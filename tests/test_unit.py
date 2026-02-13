
import os
import sys
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# 將 project root 加入 sys.path 以便 import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app, get_inference_service

# 測試用的圖片路徑
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "test", "img_0_label_6.png")

# 檢查測試圖片是否存在
if not os.path.exists(TEST_IMAGE_PATH):
    # 嘗試找任何 png
    import glob
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), "..", "test", "*.png"))
    if candidates:
        TEST_IMAGE_PATH = candidates[0]
    else:
        pytest.skip("No test image found", allow_module_level=True)

@pytest.fixture
def client():
    # 使用 context manager 啟動 lifespan (載入模型)
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_inference_service_loaded(client):
    # 若 client context 啟動成功，模型應已載入
    service = get_inference_service()
    assert service is not None
    assert service.model is not None

def test_predict_single_image(client):
    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.png", f, "image/png")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "predicted_class" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert isinstance(data["prediction"]["predicted_class"], int)

def test_predict_invalid_content_type(client):
    with open(TEST_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", f, "text/plain")}
        )
    # 預期 400 Bad Request
    assert response.status_code == 400
    assert "Content-Type" in response.json()["detail"]

def test_predict_batch(client):
    # 準備多個檔案
    files = []
    # 重複使用同一張圖片模擬批次
    content = open(TEST_IMAGE_PATH, "rb").read()
    
    files = [
        ("files", ("img1.png", content, "image/png")),
        ("files", ("img2.png", content, "image/png"))
    ]
    
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 200
    
    result = response.json()
    assert "results" in result
    assert len(result["results"]) == 2
    assert result["results"][0]["filename"] == "img1.png"
    assert "prediction" in result["results"][0]
