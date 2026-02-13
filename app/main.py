
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import os
import io
from typing import List

# 嘗試相對導入，如果在 project/ 下執行
try:
    from app.inference import get_inference_service
except ImportError:
    # 這是為了方便單獨測試 app/main.py
    from inference import get_inference_service

# 定義 lifespan 來初始化資源
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時載入模型
    model_path = os.getenv("MODEL_PATH", "model_weights.pth")
    if not os.path.exists(model_path):
        # 嘗試上一層目錄（如果是在 app/ 下執行）
        model_path = os.path.join("..", "model_weights.pth")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        get_inference_service(model_path)
    else:
        print(f"Warning: Model file not found at {model_path}. Please set MODEL_PATH env var.")
    
    yield
    # 關閉時清理資源（如果有的話）
    print("Shutting down API...")

app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="A simple API to predict handwritten digits using a CNN-Transformer model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Handwritten Digit Recognition API. Visit /docs for Swagger UI."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    上傳圖片並進行數字辨識。
    支援格式: PNG, JPG, JPEG
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File provided is not an image. Content-Type: {file.content_type}")

    try:
        # 讀取檔案內容
        contents = await file.read()
        
        # 取得全域推論服務實例
        service = get_inference_service()
        if service is None:
             raise HTTPException(status_code=500, detail="Model not initialized.")
            
        # 執行推論
        result = service.predict(contents)
        return {
            "filename": file.filename, 
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    批次上傳圖片並進行數字辨識。
    回應包含每張圖片的檔名與預測結果。
    """
    service = get_inference_service()
    if service is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    
    results = []
    for file in files:
        # 簡單檢查 Content-Type (若有)
        if file.content_type and not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": f"Invalid Content-Type: {file.content_type}"
            })
            continue
            
        try:
            contents = await file.read()
            prediction = service.predict(contents)
            results.append({
                "filename": file.filename,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
            
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
