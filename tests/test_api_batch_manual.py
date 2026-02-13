
import sys
import os
import requests

url = "http://127.0.0.1:8000/predict_batch"
img_path = "test/img_0_label_6.png"

# 確保路徑正確 (假設在 project 根目錄執行，或 tests 目錄)
if not os.path.exists(img_path):
    # 嘗試上一層
    if os.path.exists(f"../{img_path}"):
        img_path = f"../{img_path}"
    # 嘗試絕對路徑查找 test 資料夾
    elif os.path.exists("test"):
        import glob
        imgs = glob.glob("test/*.png")
        if imgs:
            img_path = imgs[0]

if not os.path.exists(img_path):
    print(f"Error: Test image not found at {img_path}")
    sys.exit(1)

print(f"Testing batch predict at {url}...")
print(f"Using image: {img_path}")

try:
    # 模擬上傳 3 張圖片
    files = [
        ("files", ("batch_1.png", open(img_path, "rb"), "image/png")),
        ("files", ("batch_2.png", open(img_path, "rb"), "image/png")),
        ("files", ("batch_3.png", open(img_path, "rb"), "image/png"))
    ]
    
    response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Response:", result)
        print(f"Batch prediction successful! Processed {len(result['results'])} images.")
    else:
        print("Failed:", response.text)
        
except requests.exceptions.ConnectionError:
    print("Connection refused. Is uvicorn running?")
except Exception as e:
    print(f"Error: {e}")
