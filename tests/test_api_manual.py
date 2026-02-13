
import sys
import os

try:
    import requests
except ImportError:
    print("requests module not found. Install it with: pip install requests")
    sys.exit(1)

def test_api():
    url = "http://127.0.0.1:8000/predict"
    
    # 尋找測試圖片
    image_path = "test/img_0_label_6.png"
    if not os.path.exists(image_path):
        # 嘗試找任何一張圖片
        import glob
        images = glob.glob("test/*.png") + glob.glob("test/*.jpg")
        if images:
            image_path = images[0]
        else:
            print("Error: No test images found in test/ directory.")
            return

    print(f"Sending POST request to {url} with image: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:", response.json())
            print("API Test Passed!")
        else:
            print("Response Content:", response.text)
            print("API Test Failed.")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure uvicorn is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # 切換到 project 根目錄（假設腳本在 tests/ 下執行）
    if os.path.basename(os.getcwd()) == "tests":
        os.chdir("..")
        
    test_api()
