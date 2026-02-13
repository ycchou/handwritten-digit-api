
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from .model import CNNTransformer

class InferenceService:
    def __init__(self, model_path: str = "model_weights.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNTransformer()
        try:
            # 嘗試載入模型權重，處理 CPU/GPU 差異
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path} on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def preprocess(self, image_data: bytes) -> torch.Tensor:
        """
        將圖片 bytes 轉換為模型輸入 tensor
        """
        try:
            image = Image.open(io.BytesIO(image_data)).convert('L')
            tensor = self.transform(image).unsqueeze(0) # [1, 1, 28, 28]
            return tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

    def predict(self, image_data: bytes):
        """
        輸入圖片 bytes，回傳預測結果 (class_id, confidence)
        """
        tensor = self.preprocess(image_data)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            # 使用 softmax 取得機率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            return {
                "predicted_class": int(predicted.item()),
                "confidence": float(confidence.item())
            }

# 全域實例，稍後在 main.py 初始化
inference_service = None

def get_inference_service(model_path="model_weights.pth"):
    global inference_service
    if inference_service is None:
        inference_service = InferenceService(model_path)
    return inference_service
