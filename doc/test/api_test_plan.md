
---
description: 手寫數字辨識 API 測試案例
---

狀態：初始為 [ ]、完成為 [x] 
注意：狀態只能在測試通過後由流程更新。 
測試類型：API 邏輯、圖片預處理、錯誤處理、模型推論

## API 端點測試 (App)

### [x] 【API 邏輯】單張圖片預測成功 (POST /predict)
測試說明1：上傳合法 PNG 圖片，應回傳預測類別與信心度
範例輸入：`test/img_0_label_6.png`
期待輸出：HTTP 200, JSON 包含 `filename`, `prediction` (含 `predicted_class`: 6)

### [x] 【API 邏輯】批次圖片預測成功 (POST /predict_batch)
測試說明2：同時上傳兩張圖片，應回傳包含兩個預測結果的列表
範例輸入：`[test/img_0_label_6.png, test/img_1_label_5.png]`
期待輸出：HTTP 200, JSON 包含 `results` 列表長度為 2，分別預測為 6 與 5

### [x] 【錯誤處理】非圖片檔案上傳 (POST /predict)
測試說明3：上傳文字檔作為圖片，應被攔截並回傳錯誤
範例輸入：`test.txt` (Content-Type: text/plain)
期待輸出：HTTP 400, detail 包含 "File provided is not an image"

### [x] 【錯誤處理】無效內容類型 (POST /predict_batch)
測試說明4：批次上傳中包含非圖片檔案，該項目應回傳錯誤訊息，但不影響其他圖片
範例輸入：`[test/img_0_label_6.png, test.txt]`
期待輸出：HTTP 200, `results` 列表長度為 2，第一項有 `prediction`，第二項有 `error`

## 模型與服務測試 (Service)

### [x] 【模型推論】模型載入與推論服務初始化
測試說明5：確認 `get_inference_service()` 能正確載入模型權重
範例輸入：`model_weights.pth` 存在
期待輸出：Service instance 非 None，Model instance 非 None，且處於 eval 模式

### [x] 【圖片預處理】圖片轉換 Tensor 形狀檢查
測試說明6：輸入圖片 bytes，預處理後 Tensor 形狀應為 [1, 1, 28, 28]
範例輸入：任意圖片 bytes
期待輸出：Tensor shape `torch.Size([1, 1, 28, 28])`
