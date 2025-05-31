# Étape 1 : Image de base
FROM python:3.10-slim

# Étape 2 : Répertoire de travail
WORKDIR /app

# Étape 3 : Copier le contenu
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --upgrade pip \
 && pip install flask==3.1.1 werkzeug==3.1.3 tensorflow pillow numpy pandas

# Étape 5 : Exposer le port Flask
EXPOSE 5000

# Étape 6 : Lancer l’API
CMD ["python", "FlaskApi.py"]
