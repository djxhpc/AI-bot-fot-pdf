import os
import pandas as pd
import ollama
import time
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 設定測試參數
MODELS = [
    "ycchen/breeze-7b-instruct-v1_0:latest",
    "cwchang/llama3-taide-lx-8b-chat-alpha1:latest",
    "gemma4:latest",
    "qwen2.5:latest",
    "llama3.2:latest"
]
K_VALUES = [1, 5, 10, 20]

# 2. 設定測試資料集與路徑 (請確保 PDF 檔案路徑正確)
TEST_DATA = [
    {
        "label": "員工守則_14229字",
        "pdf_path": r"D:\work2\pdf測試\employee_rule14229.pdf", 
        "index_dir": "indexes/employee_rule14229",
        "questions": ["請問這是甚麼公司的守則?", "第二條內容是?", "凡有下列情事之一者，不得僱用為本公司員工?", "本公司員工待遇均按到職之日起支薪，離職之日停薪。 是第幾條規則?", "公司員工工作時間每日以幾小時為原則","考績乙等區間為","記申誡者，其當年度考績不得列為?","員工非有下列情形之一者，不得強制退休?","本公司嚴禁就業場所之性騷擾行為，訂定什麼法的標準作業程序書，並在工作場所公開揭示。","最多到第幾條規則?"]

    },
    {
        "label": "大型資料集_30672字",
        "pdf_path": r"D:\work2\pdf測試\30672.pdf",
        "index_dir": "indexes/30672",
        "questions": ["請問這是甚麼的守則", "第 一 章  總  則的第十二條是? ", "本公司得視實際需要，依勞動基準法什麼規定實施彈性工時。", "員工繼續工作四小時，至少應有三十分鐘之休息。但實行輪班制或其工作有連續性或緊急性者，本公司得在工作時間內，另行調配其休息時間。是第幾條? ", "因天災、事變或突發事件，本公司認為有繼續工作之必要時，得停止第二十五條至第二十七條所定員工之例假、休假及特別休假，但應於事後多少小時內，詳述理由，報請當地主管機關核備。停止假期之工資，加倍發給，並應於事後補假休息。","員工遭遇職業傷害或罹患職業病而死亡時，本公司除給與五個月平均工資之喪葬費外，並應一次給與其遺屬四十個月平均工資之死亡補償。其遺屬受領死亡補償之順位如下：","員工適用勞動基準法退休金規定者，其請領退休金權利，自退休之次月起，因幾年間不行使而消滅。請領退休金權利不得讓與、抵銷、扣押或供擔保。","最多到第幾條規則?","勞動部中華民國一一五年一月八日勞動條1字第?號函修正","未滿?十人之事業單位，其工作規則審核準用本要點。"]
  
    }
]

def get_or_create_vector_db(data, embeddings):
    """檢查索引是否存在，若無則讀取 PDF 建立並存檔"""
    index_path = data["index_dir"]
    
    # 如果索引資料夾已存在，直接載入
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        print(f"✅ 找到現有索引，正在載入: {index_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 否則，讀取 PDF 並建立索引
    print(f"🔨 找不到索引，正在從 PDF 建立: {data['pdf_path']}")
    if not os.path.exists(data['pdf_path']):
        raise FileNotFoundError(f"找不到 PDF 檔案: {data['pdf_path']}")
        
    reader = PdfReader(data['pdf_path'])
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # 切片處理 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)
    docs = [Document(page_content=t) for t in texts]
    
    # 建立並存檔 (會產出 index.faiss 和 index.pkl)
    vector_db = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vector_db.save_local(index_path)
    print(f"💾 索引已儲存至: {index_path}")
    return vector_db

def run_test():
    results = []
    # 初始化 Embedding 模型
    print("⏳ 正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    for data in TEST_DATA:
        try:
            vector_db = get_or_create_vector_db(data, embeddings)
        except Exception as e:
            print(f"❌ 處理 {data['label']} 時出錯: {e}")
            continue
            
        for k in K_VALUES:
            for model_name in MODELS:
                for q in data['questions']:
                    print(f"🚀 [測試中] 模型: {model_name.split(':')[0]} | K: {k} | 資料集: {data['label']}")
                    
                    # 檢索
                    docs = vector_db.similarity_search(q, k=k)
                    context = "\n\n---\n\n".join([d.page_content for d in docs])
                    
                    # 呼叫 Ollama
                    start_time = time.time()
                    try:
                        response = ollama.chat(model=model_name, messages=[
                            {'role': 'system', 'content': '你是一個專業助手，請根據提供的參考內容回答問題。'},
                            {'role': 'user', 'content': f"【參考內容】:\n{context}\n\n【問題】: {q}"}
                        ])
                        answer = response['message']['content']
                    except Exception as e:
                        answer = f"錯誤: {e}"
                    
                    duration = time.time() - start_time
                    
                    # 儲存結果
                    results.append({
                        "資料集標籤": data["label"],
                        "模型名稱": model_name,
                        "K值": k,
                        "測試問題": q,
                        "AI回答": answer,
                        "生成耗時(秒)": round(duration, 2),
                        "人工評核(正確1/錯誤0)": "" 
                    })

    # 5. 確保資料夾存在並存檔
    if results:
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d-%H%M")
        
        # 先存 CSV 
        csv_name = f"RAG_backup_{timestamp}.csv"
        df.to_csv(csv_name, index=False, encoding="utf-8-sig")
        print(f"備份檔已存入: {csv_name}")
        
        # 再存 Excel
        try:
            output_name = f"RAG_結果_{timestamp}.xlsx"
            df.to_excel(output_name, index=False)
            print(f"Excel 檔已存入: {output_name}")
        except Exception as e:
            print(f"Excel 存檔失敗，但 CSV 已保留。錯誤原因: {e}")
    else:
        print("❌ 沒有產生任何測試結果")

if __name__ == "__main__":
    run_test()


