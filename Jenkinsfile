pipeline {
    agent any

  

    stages {
        stage('Clone repository') {
            steps {
                git url: 'https://github.com/ton-compte/ton-projet.git' , branch: 'main'
            }
        }

        

        

        stage('Build Docker image') {
            steps {
                bat 'docker build -t flask-api:latest .'
            }
        }

        stage('Run container') {
            steps {
                bat 'docker run -d -p 5000:5000 --name flask-api-container flask-api:latest'
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
    }
}

