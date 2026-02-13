
import os
import sys
import pytest
import torch
from fastapi.testclient import TestClient

# 將 project root 加入 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.main import app, get_inference_service

# 測試用的圖片路徑
TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test")
IMG_6 = os.path.join(TEST_DIR, "img_0_label_6.png")
IMG_5 = os.path.join(TEST_DIR, "img_1_label_5.png")

# 確保測試圖片存在
if not os.path.exists(IMG_6):
    pytest.skip("Test image img_0_label_6.png not found", allow_module_level=True)

@pytest.fixture
def client():
    # 使用 context manager 啟動 lifespan (載入模型)
    with TestClient(app) as c:
        yield c

# --- API 端點測試 (App) ---

def test_predict_single_image(client):
    """
    測試說明1：上傳合法 PNG 圖片，應回傳預測類別與信心度
    """
    with open(IMG_6, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("img_0.png", f, "image/png")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "prediction" in data
    assert data["prediction"]["predicted_class"] == 6

def test_predict_batch_images(client):
    """
    測試說明2：同時上傳兩張圖片，應回傳包含兩個預測結果的列表
    """
    if not os.path.exists(IMG_5):
        pytest.skip("Second test image not found")
        
    files = [
        ("files", ("img_0.png", open(IMG_6, "rb"), "image/png")),
        ("files", ("img_1.png", open(IMG_5, "rb"), "image/png"))
    ]
    
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 200
    
    result = response.json()
    assert "results" in result
    assert len(result["results"]) == 2
    
    # 驗證第一個結果
    assert result["results"][0]["prediction"]["predicted_class"] == 6
    # 驗證庫二個結果
    assert result["results"][1]["prediction"]["predicted_class"] == 5

def test_predict_invalid_file_type(client):
    """
    測試說明3：上傳文字檔作為圖片，應被攔截並回傳錯誤
    """
    with open(IMG_6, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", f, "text/plain")}
        )
    # 預期 400 Bad Request
    assert response.status_code == 400
    assert "File provided is not an image" in response.json()["detail"]

def test_predict_batch_mixed_content(client):
    """
    測試說明4：批次上傳中包含非圖片檔案，該項目應回傳錯誤訊息，但不影響其他圖片
    """
    files = [
        ("files", ("img_0.png", open(IMG_6, "rb"), "image/png")),
        ("files", ("test.txt", open(IMG_6, "rb"), "text/plain"))
    ]
    
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 200
    
    result = response.json()["results"]
    assert len(result) == 2
    
    # 第一張正常
    assert "prediction" in result[0]
    # 第二張報錯
    assert "error" in result[1]
    assert "Invalid Content-Type" in result[1]["error"]

# --- 模型與服務測試 (Service) ---

def test_model_initialization(client):
    """
    測試說明5：確認 get_inference_service() 能正確載入模型權重
    """
    service = get_inference_service()
    assert service is not None
    assert service.model is not None
    assert not service.model.training  # 應為 eval 模式

def test_preprocess_tensor_shape(client):
    """
    測試說明6：輸入圖片 bytes，預處理後 Tensor 形狀應為 [1, 1, 28, 28]
    """
    service = get_inference_service()
    with open(IMG_6, "rb") as f:
        img_bytes = f.read()
        
    tensor = service.preprocess(img_bytes)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 28, 28)
