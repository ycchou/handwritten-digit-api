
pipeline {
    agent any
    
    environment {
        // 設定映像檔名稱與標籤
        IMAGE_NAME = "mnist-api"
        IMAGE_TAG = "${env.BUILD_NUMBER}"
    }
    
    options {
        // 設定逾時時間，避免卡住
        timeout(time: 10, unit: 'MINUTES')
        // 保留最近 5 次建置記錄
        buildDiscarder(logRotator(numToKeepStr: '5'))
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}..."
                    // 建置 Image，同時標記為 latest
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest ."
                }
            }
        }
        
        stage('Run Unit Tests') {
            steps {
                script {
                    echo "Running tests inside container..."
                    // 啟動容器執行測試，需額外安裝測試依賴 (pytest, httpx)
                    // 注意：這裡假設 Dockerfile 已經 COPY 了程式碼與權重檔
                    sh """
                        docker run --rm ${IMAGE_NAME}:latest \
                        /bin/bash -c "pip install pytest httpx && pytest tests/test_unit.py -v"
                    """
                }
            }
        }
        
        stage('Deploy (Mock)') {
            steps {
                script {
                    echo "Deploying application..."
                    // 實際部署邏輯 (例如 docker push 或更新 K8s yaml)
                    echo "Docker push ${IMAGE_NAME}:${IMAGE_TAG} (Skipped for demo)"
                    
                    // 若使用 docker-compose 部署
                    // sh "docker-compose up -d"
                }
            }
        }
    }
    
    post {
        success {
            echo "Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed."
        }
        always {
            // 清理 Docker 資源 (非必須，視 Jenkins Agent 策略而定)
            // sh "docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true"
            cleanWs()
        }
    }
}
