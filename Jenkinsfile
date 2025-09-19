pipeline{
    agent any
    stages{
        stage('Checkout'){
            steps{
                git 'https://github.com/sahil5206/Touchless-Computing-Project-Using-Python.git'
            }
        }

        stage('Build Dcoker Image'){
            steps{
                sh 'docker build -t touchless-computing.'
            }
        }

        stage('Run GUI Container'){
            steps{
                sh ''' 
                docker run -d \ 
                --name touchless_gui_container \
                -e DISPLAY=$DISPLAY \ 
                -v /tmp/.x11-unix:/tmp/.x11-unix \
                --device /dev/video0 \
                touchless-computing
                '''
            }
        }
    }

    post{
        always{
            sh 'docker stop touchless_gui_container || true'
            sh 'docker rm touchless_gui_container || true'
        }
    }
}