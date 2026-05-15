import discord
import ollama
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 配置區 ---
TOKEN = 'XXX' #個人TOKEN
MODEL_NAME = "ycchen/breeze-7b-instruct-v1_0:latest" 
INDEX_PATH = "faiss_index_storage" 
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 設定意圖 (必須開啟 Message Content Intent)
intents = discord.Intents.default()
intents.message_content = True 
client = discord.Client(intents=intents)

# --- 核心函式：檢索與回答 ---
def get_rag_context(query):
    """從 FAISS 索引中抓取相關資料"""
    if os.path.exists(INDEX_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            # allow_dangerous_deserialization=True 是載入本機檔案必須的
            db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            # 抓取前 3 個最相關片段
            docs = db.similarity_search(query, k=3)
            return "\n\n".join([f"{d.page_content}\n[來源：{d.metadata.get('page', '未知')}]" for d in docs])
        except Exception as e:
            print(f"向量庫讀取失敗: {e}")
    return ""

def ask_ollama(prompt, context):
    system_instruction = """你現在是一位專業的企業行政助手。
            你的唯一任務是根據使用者提供的【手冊知識庫】回答每個問題。
            【強制規則】：
            1. 必須完全使用「繁體中文」（台灣習慣用語）回答。
            2. 絕對禁止使用簡體中文。
            3. 如果手冊中沒有答案，請禮貌地說找不到，不要編造。
            4. 保持專業、客觀、準確。
            5. 遇到打招呼（如：你好、早安），請用親切的語氣直接與使用者寒暄。
            6. 遇到使用者輸入完全無意義的亂碼時，請禮貌地表達你聽不懂，並引導使用者詢問手冊相關問題。
             """
     
    if "67" in prompt or "six seven" in prompt.lower():
        return "欸six seven🗣️🗣️🔥🔥🔥"

    full_prompt = f"【手冊內容】：\n{context}\n\n當前問題：{prompt}"
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': full_prompt}
        ],
        options={"temperature": 0.05}
    )
    return response['message']['content']

# --- 事件處理 ---

@client.event
async def on_ready():
    print(f"✅ Discord Bot 已上線：{client.user}")
    
    # 修正啟動檢查邏輯
    has_faiss = os.path.exists(INDEX_PATH)
    has_temp = os.path.exists("temp_context.txt")
    
    if has_faiss:
        print("📚 模式：FAISS 向量檢索 (RAG 已啟動)")
    elif has_temp:
        print("📄 模式：純文字檢索 (讀取 temp_context.txt)")
    else:
        print("⚠️ 模式：純模型回答 (未偵測到任何本地知識庫)")

@client.event
async def on_message(message):
    # 排除 Bot 自己與空訊息
    if message.author == client.user or not message.content.strip():
        return

    async with message.channel.typing():
        user_query = message.content
        context = ""
        
        # 優先權 1：檢查是否有大型向量庫
        if os.path.exists(INDEX_PATH):
            context = get_rag_context(user_query)
        
        # 優先權 2：若無向量庫，檢查是否有小型純文字檔
        elif os.path.exists("temp_context.txt"):
            try:
                with open("temp_context.txt", "r", encoding="utf-8") as f:
                    context = f.read()
            except Exception as e:
                print(f"讀取 temp_context.txt 失敗: {e}")

        # 呼叫 Ollama 進行回答
        try:
            answer = ask_ollama(user_query, context)
            
            # 處理 Discord 2000 字限制
            if len(answer) > 2000:
                for i in range(0, len(answer), 2000):
                    await message.channel.send(answer[i:i+2000])
            else:
                await message.channel.send(answer)
        except Exception as e:
            print(f"Ollama 回應出錯: {e}")
            await message.channel.send("❌ 抱歉，我現在無法處理您的請求。")

# client.run(TOKEN)
@client.event
async def on_message(message):
    if message.author == client.user or not message.content.strip():
        return

    async with message.channel.typing():
        user_query = message.content
        
        # 每次收到訊息時，都重新讀取一次最新索引
        context = ""
        if os.path.exists("faiss_index_storage"):
            context = get_rag_context(user_query)
        elif os.path.exists("temp_context.txt"):
            with open("temp_context.txt", "r", encoding="utf-8") as f:
                context = f.read()

        try:
            answer = ask_ollama(user_query, context)
            await message.channel.send(answer)
        except Exception as e:
            print(f"Error: {e}")
client.run(TOKEN)