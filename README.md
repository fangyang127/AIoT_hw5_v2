# AI vs Human 文章偵測器

Streamlit 工具，用於判別文本可能是 AI 生成或人類撰寫。提供 TF-IDF + Logistic Regression baseline，並可選 Hugging Face transformers 模型。

## 目錄結構
```
src/
  app.py        # Streamlit 入口
  models.py     # Baseline 與 transformers 模型
  utils.py      # 前處理、檔案讀取、範例文本
requirements.txt
```

## 快速開始
1. 建立並啟用虛擬環境（可選）。
2. 安裝相依：
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動介面：
   ```bash
   streamlit run src/app.py
   ```
4. 瀏覽器將自動開啟，或手動前往顯示的本機網址。

## 功能重點
- 輸入：文字框或上傳 `.txt/.docx`（限制 2 MB）。
- 模型：
  - Baseline：TF-IDF + Logistic Regression（小型示範語料）。
  - Transformers：`roberta-base-openai-detector` 或 `distilroberta-base`（可在 `get_available_hf_models` 中增減）。
- 前處理：清理空白、語言偵測（非中英則提示）、長文截斷/分段平均。
- 輸出：AI% / Human% 機率、置信度條圖與甜甜圈圖、模型資訊、耗時。
- 評估：側邊可切換範例文本快速測試。

## 自訂與部署
- 想要更強的 baseline，可將 `models.py` 中 `_seed_corpus` 改為自有標註資料並重新訓練。
- 若 transformers 模型標籤不同，可在 `map_hf_labels` 補充映射規則。
- 將專案上傳至 GitHub 後，可在 README 加上部署連結（如 Streamlit Cloud）。

## 注意
- Transformers 模型首次下載可能較久；可預先執行一次以快取。
- 範例程式為教學示範，正式場景需更大語料與嚴謹評估。***
