# AI-bot-fot-pdf
基於 Ollama模型實作的 RAG 檢索Streamlit介面助手。支援 PDF 解析、自動疊字清理、動態 K 值檢索，並整合聯發科Breeze-7B等中文模型提供精準對話。


#### 複製專案
```bash
git clone 
cd <專案資料夾名稱>
```
#### 安裝所需套件
```bash
pip install -r requirements.txt
```
#### 啟動 Ollama 並下載模型
```bash
ollama run ycchen/breeze-7b-instruct-v1_0:latest
```
#### 啟動
```bash
streamlit run 67.py
```
