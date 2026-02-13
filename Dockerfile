
# 使用 Python 3.12 slim 版本作為基礎映像檔，減少體積
FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 安裝基本工具，避免部分 pip 安裝失敗
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 特別安裝 PyTorch CPU 版本以減小 Image 大小
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 複製應用程式碼
COPY app ./app
COPY model_weights.pth .
COPY batch_predict.py .

# 設定環境變數
ENV MODEL_PATH=/app/model_weights.pth
ENV PYTHONPATH=/app

# 暴露 8000 port
EXPOSE 8000

# 啟動 FastAPI 服務
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
