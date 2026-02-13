
# 手寫數字辨識 API (Handwritten Digit Recognition API)

本專案提供一個基於 FastAPI 與 CNN-Transformer 模型的 RESTful API，可辨識上傳的手寫數字圖片 (0-9)。

## 專案結構
```
project/
├── app/
│   ├── main.py          # FastAPI 主程式
│   ├── model.py         # PyTorch 模型架構 (CNNTransformer)
│   └── inference.py     # 推論服務邏輯
├── model_weights.pth    # 預訓練模型權重
├── batch_predict.py     # 批次預測腳本
├── Dockerfile           # Docker 建置檔
├── requirements.txt     # Python 依賴清單
└── README.md            # 說明文件
```

## 1. 快速開始 (本地執行)

### 環境需求
- Python 3.10+
- 建議使用虛擬環境

### 安裝依賴
```bash
# 建立虛擬環境 (Windows)
python -m venv venv
.\venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
# 若無 GPU，建議安裝 CPU 版 PyTorch 以節省空間：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 啟動 API
```bash
python -m uvicorn app.main:app --reload
```
啟動後，API 預設運行於 `http://127.0.0.1:8000`。
- **Swagger 文件**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **預測端點**: `POST /predict`

### API 測試方式 (Postman / curl)
使用 `curl` 測試：
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@test/img_0_label_6.png"
```
預期回傳：
```json
{
  "filename": "img_0_label_6.png",
  "prediction": {
    "predicted_class": 6,
    "confidence": 0.99
  }
}
```json
{
  "filename": "img_0_label_6.png",
  "prediction": {
    "predicted_class": 6,
    "confidence": 0.99
  }
}
```

### API 介面規格 (API Reference)

#### 1. 單張圖片預測
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**: 
    - `file`: 圖片檔案 (binary)
- **Response**:
    ```json
    {
        "filename": "demo.png",
        "prediction": {
            "predicted_class": 5,
            "confidence": 0.982
        }
    }
    ```

#### 2. 批次圖片預測
- **URL**: `/predict_batch`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
    - `files`: 多個圖片檔案
- **Response**:
    ```json
    {
        "results": [
            {
                "filename": "img1.png",
                "prediction": { "predicted_class": 1, "confidence": 0.99 }
            },
            {
                "filename": "test.txt",
                "error": "Invalid Content-Type: text/plain"
            }
        ]
    }
    ```


## 2. 批次預測 (Batch Prediction)
執行 `batch_predict.py` 可對 `test/` 資料夾內的所有圖片進行推論，結果將輸出至 `result.csv`。

```bash
python batch_predict.py
```
執行完畢後請查看專案根目錄下的 `result.csv`。

## 3. Docker 部署
本專案包含 `Dockerfile`，可將 API 封裝為 Docker Image。

### 建置 Image
```bash
docker build -t mnist-api .
```

### 啟動 Container
```bash
docker run -p 8000:8000 mnist-api
```
服務將運行在 `localhost:8000`。

---

## 問題回答

### Q2. 為何使用這樣的架構？
本專案採用 **FastAPI + PyTorch (CNN-Transformer)** 架構，原因如下：
1. **模型效能**：CNN 擅長提取影像局部特徵 (如筆畫邊緣)，Transformer 擅長捕捉全局關係 (筆畫順序與結構)，兩者結合能在手寫辨識任務上達到高準確率。
2. **API 效能**：FastAPI 基於 ASGI 標準，具備極高的併發處理能力，且內建自動化文件 (Swagger UI)，大幅降低前後端對接成本。
3. **部署彈性**：程式碼模組化設計並容器化 (Docker)，確保在任何環境 (本地、雲端、K8s) 皆能一致運行，符合「開服測試人多不能爆掉」的高可用需求。

### Q3. 該如何跟 PM 說要怎麼做給客戶，要怎麼跟他們說以及怎麼做？

**溝通策略：**

**1. 對 PM 說 (強調穩定與時程)：**
> 「我已經將模型打包成標準的 API 服務，並完成了 Docker 容器化。這意味著無論客戶端的伺服器環境為何，我們都能確保服務『隨插即用』，不會因為環境設定不同而無法執行。同時 API 支援並發處理，能應對開服時的高流量。我還準備了批次預測腳本與詳細文件，方便客戶驗收與整合。」

**2. 對客戶說 (強調易用與交付)：**
> 「我們為貴公司提供了封裝好的 Docker Image 與 RESTful API。
> - **部署簡單**：只需執行一行 Docker 指令即可啟動服務。
> - **整合容易**：API 符合標準 REST 規範，並附帶互動式文件 (Swagger)，您的工程師可以直接在網頁上測試。
> - **驗收方便**：附帶批次預測工具，可直接對大量圖片進行驗證。」

**具體執行步驟：**
1. 將專案推送到 GitHub/GitLab Repo。
2. 提供 Docker Image (可推至 Docker Hub 或 ECR)。
3. 交付 `README.md` 與 API URL 給客戶端工程師。

---

## 4. 加分項 (Bonus Features)

本專案已實作以下加分功能：

### 4.1 批次預測 API (`/predict_batch`)
除了腳本外，API 新增了 `POST /predict_batch` 端點，支援同時上傳多張圖片並回傳結果列表。
**測試指令** (需安裝 `httpx`):
```python
import httpx
files = [('files', open('test/img_0.png', 'rb')), ('files', open('test/img_1.png', 'rb'))]
r = httpx.post("http://localhost:8000/predict_batch", files=files)
print(r.json())
```

### 4.2 Unit Test
使用 `pytest` 與 `FastAPI TestClient` 撰寫單元測試，覆蓋模型載入、預測邏輯與錯誤處理。
**執行測試**：
```bash
pip install pytest httpx
pytest tests/test_unit.py
```

### 4.3 Docker Compose
提供 `docker-compose.yml`，可一鍵啟動 API 服務並掛載測試資料夾。
**啟動服務**：
```bash
docker-compose up --build -d
```

### 4.4 CI/CD Pipeline
提供 `Jenkinsfile`，定義了從 Build Image -> Test -> Deploy 的自動化流程。

### 4.5 雲端部署指引 (Cloud Deployment)

#### AWS (ECS Fargate)
1. **Push Image**: 將 Docker Image 推送至 Amazon ECR。
2. **Create Task Definition**: 設定 CPU/Memory 與 Port 8000。
3. **Run Service**: 選擇 Fargate Launch Type，設定 Auto Scaling (基於 CPU 使用率)。
4. **Load Balancer**: 設定 Application Load Balancer (ALB) 轉發流量至 Container。

#### GCP (Cloud Run)
1. **Push Image**: 將 Docker Image 推送至 Google Container Registry (GCR)。
2. **Deploy**:
   ```bash
   gcloud run deploy mnist-api --image gcr.io/[PROJECT-ID]/mnist-api --platform managed
   ```
3. **Auto Scaling**: Cloud Run 會根據請求量自動擴展實例數至 0 (節省成本) 或更多。

#### Azure (Container Instances)
1. **Push Image**: 將 Docker Image 推送至 Azure Container Registry (ACR)。
2. **Deploy**:
   ```bash
   az container create --resource-group myResourceGroup --name mnist-api --image [ACR-NAME].azurecr.io/mnist-api --dns-name-label mnist-api --ports 8000
   ```

