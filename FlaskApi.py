import os
import uuid
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.utils import to_categorical, img_to_array, load_img
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from werkzeug.utils import secure_filename
import io
from PIL import Image

# --- Config ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Classifier Model (for /predict endpoint if path provided) ---
classifier_model = None
CLASSIFIER_MODEL_PATH = 'path/to/your/classifier_model.h5'  # <--- IMPORTANT: UPDATE THIS PATH!

if os.path.exists(CLASSIFIER_MODEL_PATH):
    try:
        classifier_model = load_model(CLASSIFIER_MODEL_PATH)
        print("Classifier model loaded successfully for general use and /predict.")
    except Exception as e:
        print(f"Error loading classifier model from {CLASSIFIER_MODEL_PATH}: {e}")
        classifier_model = None
else:
    print(f"Warning: Classifier model not found at {CLASSIFIER_MODEL_PATH}. "
          "The /predict endpoint might fail if a classifier is not provided via path. "
          "The /train endpoint will require a classifier model upload.")


# --- Attack Functions (No changes here, assuming they are correct) ---
def fgsm_attack(image, label, model, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([label])
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
    gradient = tape.gradient(loss, image_tensor)
    if gradient is None:
        raise ValueError("Gradient is None. FGSM attack failed.")
    signed_grad = tf.sign(gradient)
    adv_image = image_tensor + epsilon * signed_grad
    return tf.clip_by_value(adv_image[0], 0, 1).numpy()


def pgd_attack(image, label, model, epsilon=0.03, alpha=0.007, iterations=10):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. PGD attack failed.")
        adv_image = adv_image + alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def bim_attack(image, label, model, epsilon=0.03, alpha=0.005, iterations=10):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. BIM attack failed.")
        adv_image = adv_image + alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def mim_attack(image, label, model, epsilon=0.03, alpha=0.005, iterations=10, decay=1.0):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    momentum = tf.zeros_like(image, dtype=tf.float32)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. MIM attack failed.")
        gradient = gradient / (tf.reduce_mean(tf.abs(gradient)) + 1e-8)
        momentum = decay * momentum + gradient
        adv_image = adv_image + alpha * tf.sign(momentum)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def tanh_space(x):
    return tf.tanh(x) / 2 + 0.5


def cw_attack(image, label, model, confidence=10, learning_rate=0.01, max_iterations=50):
    image_tensor = tf.cast(image, tf.float32)
    initial_w = tf.atanh(tf.clip_by_value(image_tensor * 2 - 1, -1 + 1e-6, 1 - 1e-6))
    w = tf.Variable(initial_w, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate)
    for i in range(max_iterations):
        with tf.GradientTape() as tape:
            adv_image = tanh_space(w)
            perturbation = adv_image - image_tensor
            loss_l2 = tf.reduce_sum(tf.square(perturbation))
            preds = model(adv_image[np.newaxis, ...])
            target_one_hot = tf.one_hot([label], preds.shape[-1])
            real = tf.reduce_sum(preds * target_one_hot, axis=1)
            other = tf.reduce_max(preds * (1 - target_one_hot) - (target_one_hot * 10000.0), axis=1)
            loss_misclassification = tf.maximum(0.0, other - real + confidence)
            loss = tf.reduce_sum(loss_l2) + tf.reduce_sum(loss_misclassification)
        gradients = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradients, [w]))
    return tf.clip_by_value(tanh_space(w).numpy(), 0, 1)


# --- Utility Functions ---

def load_images_from_folder(folder_path, target_size=(128, 128)):
    images, labels = [], []
    for label_name in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_name)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(label_path, file)
                    try:
                        img = load_img(img_path, target_size=target_size)
                        img_array = img_to_array(img) / 255.0
                        if img_array.shape[-1] == 4:
                            img_array = img_array[..., :3]
                        elif img_array.shape[-1] == 1:
                            img_array = np.concatenate([img_array] * 3, axis=-1)
                        images.append(img_array)
                        labels.append(label_name)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)


def label_to_index(labels):
    mapping = {
        'propre': 0, 'FGSM': 1, 'BIM': 2, 'MIM': 3, 'PGD': 4, 'CW': 5
    }
    for i in range(6):
        if i not in mapping:
            mapping[i] = i
    indexes = []
    for l in labels:
        if l in mapping:
            indexes.append(mapping[l])
        elif isinstance(l, (int, np.integer)) and l in mapping:
            indexes.append(mapping[l])
        else:
            raise ValueError(f"Unknown label: '{l}' (type: {type(l)}). "
                             "Please ensure labels are 'propre', 'FGSM', 'BIM', 'MIM', 'PGD', 'CW', or corresponding integers 0-5.")
    return np.array(indexes)


def build_detector_model(input_shape, num_classes=6):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_adversarial_dataset(clean_images, clean_labels, classifier_model_instance, output_path):
    if classifier_model_instance is None:
        raise ValueError("Classifier model instance is None. Required for attack generation.")

    # Ensure images have 3 channels (if they were grayscale) and are float32
    processed_clean_images = []
    for img in clean_images:
        if img.shape[-1] == 1:  # Grayscale
            processed_clean_images.append(np.concatenate([img] * 3, axis=-1))
        elif img.shape[-1] == 4:  # RGBA
            processed_clean_images.append(img[..., :3])
        else:
            processed_clean_images.append(img)
    clean_images = np.array(processed_clean_images, dtype=np.float32)  # Ensure float32

    adv_data = []
    adv_data += [('propre', img) for img in clean_images]

    attack_types = [
        ('FGSM', fgsm_attack),
        ('BIM', bim_attack),
        ('MIM', mim_attack),
        ('PGD', pgd_attack),
        ('CW', cw_attack)
    ]

    for attack_name, attack_fn in attack_types:
        print(f"Generating {attack_name} attacks...")
        for i, (img, lbl) in enumerate(zip(clean_images, clean_labels)):
            # Add print statements to see what's being passed to attack functions
            print(f"--- Calling {attack_name} for image {i} with label {lbl} ---")
            print(f"  Input image shape: {img.shape}, dtype: {img.dtype}")
            try:
                adv_img = attack_fn(img, lbl, classifier_model_instance)
                adv_data.append((attack_name, adv_img))
            except Exception as e:
                print(f"Error generating {attack_name} attack for label {lbl} (image {i}): {e}")
                # CONTINUE even if one image fails, to see if others pass
                continue

    out_df = pd.DataFrame({
        'label': [lbl for lbl, _ in adv_data],
        'image': [img for _, img in adv_data]
    })
    out_df.to_parquet(output_path)


def train_detector(dataset_path, save_path):
    df = pd.read_parquet(dataset_path)
    # BEFORE np.stack, check types and shapes
    print(f"--- Debugging `train_detector` data preparation ---")
    print(f"DataFrame 'image' column head type: {type(df['image'].iloc[0])}")
    if isinstance(df['image'].iloc[0], np.ndarray):
        print(f"First image array shape: {df['image'].iloc[0].shape}, dtype: {df['image'].iloc[0].dtype}")
    else:
        print(f"First image data (NOT numpy array): {df['image'].iloc[0]}")  # Print the problematic data

    x = np.stack(df['image'].to_numpy())
    y = to_categorical(label_to_index(df['label'].to_numpy()), num_classes=6)

    model = build_detector_model(x.shape[1:])
    print(f"Training detector model with input shape: {x.shape[1:]}")
    model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save(save_path)
    print(f"Detector model saved to: {save_path}")


# --- Flask Routes ---

@app.route('/train', methods=['POST'])
def train_endpoint():
    if 'dataset' not in request.files:
        return jsonify({'error': 'Missing dataset file. Please upload a .zip or .parquet file.'}), 400
    if 'classifier_model_file' not in request.files:
        return jsonify({'error': 'Missing classifier model file. Please upload a .h5 Keras model.'}), 400

    dataset_file = request.files['dataset']
    classifier_model_file = request.files['classifier_model_file']

    if dataset_file.filename == '':
        return jsonify({'error': 'No selected dataset file.'}), 400
    if classifier_model_file.filename == '':
        return jsonify({'error': 'No selected classifier model file.'}), 400

    dataset_filename = secure_filename(dataset_file.filename)
    classifier_filename = secure_filename(classifier_model_file.filename)

    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
    classifier_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], classifier_filename)

    dataset_file.save(dataset_path)
    classifier_model_file.save(classifier_upload_path)
    print(f"Dataset saved to: {dataset_path}")
    print(f"Classifier model saved temporarily to: {classifier_upload_path}")

    adv_dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], f"adv_dataset_{uuid.uuid4()}.parquet")
    detector_model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detector_model_{uuid.uuid4()}.h5")

    loaded_classifier_model = None
    try:
        loaded_classifier_model = load_model(classifier_upload_path)
        print("Classifier model loaded successfully from upload.")

        images = None
        labels = None
        target_image_size = (128, 128)

        if dataset_filename.lower().endswith('.zip'):
            print("Processing ZIP file...")
            extract_path = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_{uuid.uuid4()}")
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Dataset extracted to: {extract_path}")
            images, labels = load_images_from_folder(extract_path, target_size=target_image_size)
            numeric_labels = label_to_index(labels)
            # shutil.rmtree(extract_path) # Uncomment to clean up extracted folder

        elif dataset_filename.lower().endswith('.parquet'):
            print("Processing Parquet file...")
            df = pd.read_parquet(dataset_path)
            loaded_images = []
            loaded_labels = []

            for index, row in df.iterrows():
                image_data = row['image']
                label = row['label']

                # Debugging print: Check the type of `image_data` right after reading from DataFrame
                if index < 5:  # Print for first few rows to avoid spamming console
                    print(f"Parquet row {index}: image_data type: {type(image_data)}")
                    if isinstance(image_data, dict):
                        print(f"Parquet row {index}: image_data keys: {image_data.keys()}")
                    elif isinstance(image_data, np.ndarray):
                        print(f"Parquet row {index}: image_data shape: {image_data.shape}, dtype: {image_data.dtype}")
                    else:
                        print(f"Parquet row {index}: Unexpected image_data format: {image_data}")

                if isinstance(image_data, dict) and 'bytes' in image_data:
                    try:
                        img_bytes = image_data['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                        img = img.resize(target_image_size)
                        img_array = np.array(img) / 255.0

                        if img_array.shape[-1] == 4:
                            img_array = img_array[..., :3]
                        elif img_array.ndim == 2:
                            img_array = np.expand_dims(img_array, axis=-1)
                            img_array = np.concatenate([img_array] * 3, axis=-1)

                        loaded_images.append(img_array)
                        loaded_labels.append(label)

                    except Exception as e:
                        print(f"Error processing image from parquet row {index} (bytes dict): {e}")
                        continue

                elif isinstance(image_data, np.ndarray):
                    processed_img = image_data.astype(np.float32)
                    if processed_img.max() > 1.0:
                        processed_img = processed_img / 255.0

                    if processed_img.shape[-1] == 4:
                        processed_img = processed_img[..., :3]
                    elif processed_img.ndim == 2:
                        processed_img = np.expand_dims(processed_img, axis=-1)
                        processed_img = np.concatenate([processed_img] * 3, axis=-1)

                    # Ensure resize for numpy arrays if they aren't already target_size
                    if processed_img.shape[0] != target_image_size[0] or processed_img.shape[1] != target_image_size[1]:
                        # Convert to PIL Image for resize, then back to numpy
                        pil_img = Image.fromarray(
                            (processed_img * 255).astype(np.uint8))  # Convert back to uint8 for PIL
                        pil_img = pil_img.resize(target_image_size)
                        processed_img = np.array(pil_img) / 255.0  # Normalize again
                        # Re-check channels after PIL conversion if needed (PIL might preserve/change)
                        if processed_img.shape[-1] == 4:
                            processed_img = processed_img[..., :3]
                        elif processed_img.ndim == 2:
                            processed_img = np.concatenate([np.expand_dims(processed_img, axis=-1)] * 3, axis=-1)

                    loaded_images.append(processed_img)
                    loaded_labels.append(label)

                else:
                    print(
                        f"Unsupported image data format in parquet row {index}: {type(image_data)}. Data: {image_data}")
                    continue

            if not loaded_images:
                raise ValueError("No valid images could be loaded from the Parquet file. Check format.")

            images = np.stack(loaded_images)
            labels = np.array(loaded_labels)
            numeric_labels = label_to_index(labels)
            print("Dataset loaded from Parquet file.")

        else:
            return jsonify({'error': 'Unsupported dataset file type. Please upload a .zip or .parquet dataset.'}), 400

        if images is None or len(images) == 0:
            return jsonify({'error': 'No images found or processed from the dataset.'}), 400

        # Debugging print: Check final `images` array before generating adversarial dataset
        print(f"--- Final `images` array before generate_adversarial_dataset ---")
        print(f"  Shape: {images.shape}, Dtype: {images.dtype}")
        print(f"  Min value: {np.min(images)}, Max value: {np.max(images)}")
        print(f"  First image pixel values (top-left): {images[0, 0, 0, :]}")

        print("Generating adversarial dataset...")
        generate_adversarial_dataset(images, numeric_labels, loaded_classifier_model, adv_dataset_path)
        print(f"Adversarial dataset generated and saved to: {adv_dataset_path}")

        print("Starting detector model training...")
        train_detector(adv_dataset_path, detector_model_path)
        print(f"Detector model trained and saved to: {detector_model_path}")

        return jsonify({
            'message': 'Training completed successfully',
            'adversarial_dataset_path': adv_dataset_path,
            'detector_model_path': detector_model_path
        }), 200

    except ValueError as ve:
        print(f"ValueError during training: {ve}")
        return jsonify({'error': f'Data processing or model configuration error: {str(ve)}'}), 400
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        return jsonify({'error': f'An internal server error occurred during training: {str(e)}'}), 500
    finally:
        if os.path.exists(classifier_upload_path):
            os.remove(classifier_upload_path)
            print(f"Cleaned up temporary classifier file: {classifier_upload_path}")
        # Add cleanup for dataset_path if needed
        # if os.path.exists(dataset_path):
        #     os.remove(dataset_path)


@app.route('/predict', methods=['POST'])
def predict_with_paths():
    try:
        data = request.json
        detector_path = data.get('detector_path')
        classifier_path = data.get('classifier_path')
        image_path = data.get('image_path')

        if not (detector_path and classifier_path and image_path):
            return jsonify({'error': 'Missing one or more paths (detector_path, classifier_path, image_path)'}), 400

        if not os.path.exists(detector_path):
            return jsonify({'error': f'Detector model not found at {detector_path}'}), 404
        if not os.path.exists(classifier_path):
            return jsonify({'error': f'Classifier model not found at {classifier_path}'}), 404
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found at {image_path}'}), 404

        detector = load_model(detector_path)
        classifier = load_model(classifier_path)

        img = load_img(image_path, target_size=(128, 128))
        img_arr = img_to_array(img) / 255.0

        if img_arr.shape[-1] == 4:
            img_arr = img_arr[..., :3]
        elif img_arr.shape[-1] == 1:
            img_arr = np.concatenate([img_arr] * 3, axis=-1)

        img_arr = np.expand_dims(img_arr, axis=0)

        pred_detector = detector.predict(img_arr)
        pred_detector_class = np.argmax(pred_detector, axis=1)[0]

        attack_labels = ['propre', 'FGSM', 'BIM', 'MIM', 'PGD', 'CW']
        detected_attack_type = attack_labels[pred_detector_class]

        if detected_attack_type == 'propre':
            cls_pred = classifier.predict(img_arr)
            cls_class = np.argmax(cls_pred, axis=1)[0]
            return jsonify({'result': 'propre', 'classified_label': int(cls_class)})
        else:
            return jsonify({'result': detected_attack_type})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')