
import os
import csv
import glob
import sys
from app.inference import InferenceService

# 配置
MODEL_PATH = "model_weights.pth"
TEST_DIR = "test"
OUTPUT_FILE = "result.csv"

def predict_batch():
    # 檢查路徑
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        sys.exit(1)
    
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory '{TEST_DIR}' not found.")
        sys.exit(1)

    print("Initializing model...")
    try:
        service = InferenceService(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 搜尋圖片
    image_files = glob.glob(os.path.join(TEST_DIR, "*.png")) + \
                  glob.glob(os.path.join(TEST_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(TEST_DIR, "*.jpeg"))
    
    if not image_files:
        print(f"No images found in '{TEST_DIR}'.")
        return

    print(f"Found {len(image_files)} images. Starting prediction...")

    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted_class"])
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                try:
                    with open(img_path, 'rb') as img_f:
                        image_bytes = img_f.read()
                    
                    # 執行推論
                    result = service.predict(image_bytes)
                    pred_class = result["predicted_class"]
                    
                    writer.writerow([filename, pred_class])
                    print(f"Processed: {filename} -> {pred_class}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        print(f"Batch prediction complete. Results saved to '{OUTPUT_FILE}'.")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    predict_batch()
