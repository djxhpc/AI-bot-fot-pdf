import streamlit as st
from ollama import chat
from pypdf import PdfReader
import ollama
from datasets import load_dataset
import pandas as pd
import re
from langchain_core.documents import Document
###新增模型  llama3-taiwan-70b
# 1. 頁面配置
st.set_page_config(page_title="AI Assistant", layout="wide")

# 2. CSS 注入：優化對話框氣泡與佈局
st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none;
    }
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div[data-testid="stRadio"] label div[role="presentation"] {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        margin: 0px !important;
        color: #E0E0E0 !important;
        cursor: pointer;
        border: none !important;
        transition: 0.2s;
        width: 100% !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 14px !important;
        margin: 0 !important;
        opacity: 1 !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'manual_context' not in st.session_state:
    st.session_state['manual_context'] = ""
if 'temp_text' not in st.session_state:
    st.session_state['temp_text'] = ""

# 4. 側邊欄導覽選單
with st.sidebar:
    st.markdown('<p class="sidebar-header">對話</p>', unsafe_allow_html=True)
    
    menu_options = ["手冊解析與校對", "正式資料庫管理", "AI對話機器人"]
    page_selection = st.radio("選單", options=menu_options, index=0)
    
    if "手冊解析與校對" in page_selection:
        page = "手冊解析與校對"
    elif "正式資料庫管理" in page_selection:
        page = "正式資料庫管理"
    else:
        page = "AI對話機器人"

    st.divider()
    st.header("⚙️ 模型配置")
    
    try:
        response = ollama.list()
        available_models = [m.model for m in response.models]
        if available_models:
            default_model = "ycchen/breeze-7b-instruct-v1_0:latest" if "ycchen/breeze-7b-instruct-v1_0:latest" in available_models else available_models[0]
            target_model = st.selectbox("請選擇 Ollama 模型", options=available_models, index=available_models.index(default_model))
            st.success(f"✅ 已偵測到 {len(available_models)} 個模型")
        else:
            st.warning("找不到模型，請確認 ollama list 是否有內容")
            target_model = st.text_input("手動輸入模型", value="qwen2.5")
    except Exception as e:
        st.error(f"連線失敗：{e}")
        target_model = st.text_input("手動輸入模型", value="qwen2.5")

    st.subheader("⚙️ 檢索與重排配置")
    retrieval_k = st.slider("最終提供給 AI 的片段數量 (K)", min_value=1, max_value=30, value=10, help="混合搜尋後經由 Rerank 篩選出的最終菁英片段數量。")
    st.session_state['dynamic_k'] = retrieval_k

    st.header("🧹 系統維護")
    if st.button("🗑️ 清除快取紀錄"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("快取已完全清除！")
        st.rerun()

@st.cache_data
def get_test_dataset():
    dataset = load_dataset("MediaTek-Research/TCEval-v2", "drcd", split='test')
    return pd.DataFrame(dataset)

def clean_duplicated_text(text):
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u3000', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([\u4e00-\u9fa5])\1+', r'\1', text)
    for _ in range(3):
        text = re.sub(r'([\u4e00-\u9fa5]{2,10})\1', r'\1', text)
        text = re.sub(r'([\u4e00-\u9fa5]{2,10})\s\1', r'\1', text)
    text = re.sub(r'([\u4e00-\u9fa5])\s\1', r'\1', text)
    return text.strip()

# --- 1. 載入 Embedding 與 Reranker 模型 ---
@st.cache_resource
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings  # ✅ 完全修正拼字
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)      # ✅ 完全修正拼字

@st.cache_resource
def get_reranker_model():
    from sentence_transformers import CrossEncoder
    # 使用支援繁體中文的多語言重排模型 bge-reranker-v2-m3
    model_name = "BAAI/bge-reranker-v2-m3"
    return CrossEncoder(model_name)

# --- 2. 摘要索引建庫（Summary Index + MultiVector Retriever）---
def _detect_doc_meta(docs):
    """從文件前段自動偵測公司名稱與文件類型，回傳 global_meta_info 字串。"""
    sample_text = "".join([doc.page_content for doc in docs[:3]]) if docs else ""
    company_match = re.search(r"([\u4e00-\u9fa5]{2,20}(?:股份|有限|科技|集團)公司)", sample_text)
    title_match   = re.search(r"(工作規則|員工手冊|管理辦法|會議記錄|公文|合約書|契約)", sample_text)
    if company_match and title_match:
        return f"【來源文件】：{company_match.group(1)} - {title_match.group(1)}"
    elif company_match:
        return f"【來源公司】：{company_match.group(1)}"
    elif title_match:
        return f"【文件類型】：{title_match.group(1)}"
    return ""

def _split_into_sections(docs):
    """
    將文件切成語意完整的「段落」作為原文單元。
    使用較大的 chunk_size（600字）確保每段具備足夠語意，
    不再用極小 chunk 破壞上下文。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    return splitter.split_documents(docs)

def _generate_summary_for_section(section_text: str, model_name: str) -> str:
    """
    呼叫本地 Ollama，對單一原文段落生成繁中摘要（精華版）。
    摘要僅用於向量索引，不會直接給 LLM 作答。
    """
    prompt = (
        "請用繁體中文，以 2～4 句話精準摘要以下段落的核心重點，"
        "不要補充段落以外的任何資訊：\n\n"
        f"{section_text}"
    )
    try:
        resp = chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 2000},
        )
        return resp.message.content.strip()
    except Exception:
        # 摘要失敗時，退回使用原文前 200 字作為索引
        return section_text[:200]

def build_vector_store(docs, summarize_model: str = ""):
    """
    摘要索引建庫（Summary Index）：
      1. 將原文切成語意完整的段落（原文單元）
      2. 對每段呼叫 LLM 生成摘要（精華版）
      3. 以摘要的 Embedding 建立 FAISS 向量索引
      4. 同時保留「摘要 → 原文」的對照表（doc_id mapping）
      5. 同步對摘要建立 BM25 關鍵字索引

    檢索時：搜尋摘要 → 取出對應原文 → 送給 LLM 回答
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    import uuid

    global_meta_info = _detect_doc_meta(docs)
    sections = _split_into_sections(docs)

    # 注入來源標籤到原文 metadata
    for sec in sections:
        sec.metadata["doc_context"] = global_meta_info if global_meta_info else "通用文本"

    # 對每段生成摘要，並建立 doc_id 對照表
    summary_docs = []   # 用於建立向量索引（內容=摘要）
    id_to_full = {}     # doc_id → 原文 Document

    total = len(sections)
    progress_bar = st.progress(0, text="正在生成段落摘要索引...")

    for i, sec in enumerate(sections):
        doc_id = str(uuid.uuid4())

        # 生成摘要
        summary_text = _generate_summary_for_section(sec.page_content, summarize_model)
        if global_meta_info:
            summary_text = f"{global_meta_info}\n{summary_text}"

        # 摘要 Document（帶 doc_id，用於向量庫）
        summary_doc = Document(
            page_content=summary_text,
            metadata={**sec.metadata, "doc_id": doc_id, "source_type": "summary"},
        )
        summary_docs.append(summary_doc)

        # 原文 Document（帶 doc_id，用於回傳給 LLM）
        sec.metadata["doc_id"] = doc_id
        id_to_full[doc_id] = sec

        progress_bar.progress(
            (i + 1) / total,
            text=f"正在生成摘要索引… ({i+1}/{total})"
        )

    progress_bar.empty()

    with st.spinner("正在建立 FAISS 摘要向量庫 + BM25 摘要關鍵字庫..."):
        vector_db      = FAISS.from_documents(summary_docs, get_embedding_model())
        bm25_retriever = BM25Retriever.from_documents(summary_docs)

    st.success(f"✅ 摘要索引建庫完成！共 {total} 個段落，每段均有對應摘要。")

    return {
        "vector_db":      vector_db,
        "bm25_retriever": bm25_retriever,
        "id_to_full":     id_to_full,      # 摘要 doc_id → 原文
    }

# --- 3. 摘要索引檢索：搜尋摘要 → 取出原文 → Rerank → 回傳 ---
def get_relevant_context(query, hybrid_db, k_value):
    """
    檢索流程：
      1. FAISS 搜尋摘要向量庫（Dense）
      2. BM25 搜尋摘要關鍵字庫（Sparse）
      3. 去重後，透過 doc_id 對照表取出對應「原文」
      4. 以原文送入 Reranker 精排（讓 Reranker 看完整原文，評分更準）
      5. Score Threshold 過濾低相關片段，防止 LLM 胡亂回答
      6. 回傳原文拼接文本 + 帶分數列表（供 Debug）
    """
    vector_db      = hybrid_db["vector_db"]
    bm25_retriever = hybrid_db["bm25_retriever"]
    id_to_full     = hybrid_db.get("id_to_full", {})

    SCORE_THRESHOLD = 0.3   # Rerank 分數低於此值視為「知識庫無相關內容」

    candidate_k = max(k_value * 3, 15)

    # 1. 搜尋摘要向量庫（Dense）
    dense_summary_docs = vector_db.similarity_search(query, k=candidate_k)

    # 2. 搜尋摘要 BM25 庫（Sparse）
    bm25_retriever.k = candidate_k
    sparse_summary_docs = bm25_retriever.invoke(query)

    # 3. 去重（以 doc_id 為準），並取出對應原文
    seen_ids = set()
    full_text_candidates = []
    for summary_doc in dense_summary_docs + sparse_summary_docs:
        doc_id = summary_doc.metadata.get("doc_id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            # 優先取原文；若對照表遺失則退回使用摘要本身
            full_doc = id_to_full.get(doc_id, summary_doc)
            full_text_candidates.append(full_doc)

    if not full_text_candidates:
        return "__NO_RELEVANT_CONTEXT__", []

    # 4. Reranker 以原文重新精排（比用摘要排更準確）
    reranker = get_reranker_model()
    pairs  = [[query, doc.page_content] for doc in full_text_candidates]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, full_text_candidates), key=lambda x: x[0], reverse=True)

    # 5. Score Threshold：過濾低相關原文
    top_k_scored = [
        (score, doc) for score, doc in scored_docs[:k_value]
        if score >= SCORE_THRESHOLD
    ]

    if not top_k_scored:
        return "__NO_RELEVANT_CONTEXT__", []

    st.sidebar.caption(
        f"⚡ 摘要索引檢索完成！"
        f"初篩 {len(full_text_candidates)} 段原文 ➡️ Rerank 精選 {len(top_k_scored)} 段"
    )

    final_docs   = [doc for _, doc in top_k_scored]
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
    return context_text, top_k_scored


# --- 5. 頁面邏輯切換 ---

# --- 頁面一：手冊解析 ---
if page == "手冊解析與校對":
    st.title("📄 知識庫管理")
    tab_manual, tab_testset = st.tabs(["📁 上傳手冊 (PDF)", "🧪 測試資料集 (DRCD)"])

    with tab_manual:
        st.subheader("PDF 手冊解析")
        uploaded_file = st.file_uploader("選擇 PDF 文件", type="pdf", key="manual_uploader")
        
        if uploaded_file:
            if st.button("🔍 開始提取文本"):
                with st.spinner("正在讀取 PDF..."):
                    reader = PdfReader(uploaded_file)
                    documents = []
                    for i, p in enumerate(reader.pages):
                        t = p.extract_text()
                        if t:
                            clean_text = t.strip()
                            documents.append(Document(page_content=clean_text, metadata={}))  
                    st.session_state['temp_docs'] = documents 
                    st.session_state['text_area_input'] = "\n\n".join([doc.page_content for doc in documents])
                    st.success(f"解析成功！共 {len(reader.pages)} 頁。")

    with tab_testset:
        st.subheader("MediaTek-Research TCEval-v2")
        df_test = get_test_dataset()
        if "test_notes" not in st.session_state:
            st.session_state.test_notes = df_test.assign(備註="")

        edited_df = st.data_editor(
            st.session_state.test_notes,
            column_config={
                "context": st.column_config.TextColumn("文章內容", width="medium"),
                "question": "問題",
                "answers": "標準答案",
                "備註": st.column_config.TextColumn("我的註解", help="雙擊即可編輯筆記")
            },
            disabled=["context", "question", "answers"],
            hide_index=True,
            height=300,
            key="data_editor_test"
        )
        st.divider()
        st.subheader("🚀 知識庫批量導入")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_row = st.selectbox("選擇單一資料導入：", range(len(df_test)))
            if st.button("單筆導入"):
                target_data = df_test.iloc[selected_row]
                possible_keys = ['context', 'paragraph', 'text', 'content']
                found_content = next((target_data[k] for k in possible_keys if k in target_data), "")
                if found_content:
                    st.session_state['temp_text'] = found_content
                    st.session_state['manual_context'] = found_content
                    st.success(f"✅ 已導入第 {selected_row} 筆！")
                    st.rerun()

        with col2:
            st.write("一次導入整份資料集")
            if st.button("🔥 全部導入 (批次模式)"):
                with st.spinner("正在合併所有資料內容..."):
                    possible_keys = ['context', 'paragraph', 'text', 'content']
                    found_key = next((k for k in df_test.columns if k in possible_keys), None)
                    if found_key:
                        all_contexts = df_test[found_key].drop_duplicates().tolist()
                        combined_text = "\n\n--- 資料邊界 ---\n\n".join(all_contexts)
                        st.session_state['temp_text'] = combined_text
                        st.session_state['manual_context'] = combined_text
                        st.session_state['text_area_input'] = combined_text
                        st.success(f"✅ 已成功導入全部 {len(all_contexts)} 篇不重複文章！")
                        st.rerun()
                    else:
                        st.error("❌ 找不到可合併的欄位。")

    st.divider()
    st.subheader("📝 知識庫內容確認")
    c1, c2 = st.columns([4, 1]) 

    with c2:
        if st.button("🧹 執行疊字清理", help="針對 PDF 產生的重複字元進行過濾"):
            current_text = st.session_state.get('text_area_input', "")
            if current_text:
                with st.spinner("正在清理疊字..."):
                    cleaned = clean_duplicated_text(current_text)
                    st.session_state['text_area_input'] = cleaned
                    st.success("清理完成！")
                    st.rerun()
            else:
                st.warning("暫存區無內容可處理。")
    with c1:
        display_val = st.session_state.get('text_area_input', st.session_state.get('temp_text', ""))
        if not display_val:
            st.warning("⚠️ 目前暫存區無資料，請先上傳 PDF 或從測試集導入。")
        else:
            st.info(f"📊 目前暫存內容字數：{len(display_val)} 字")

    edited_text = st.text_area(
        "校對編輯區：(確定內容後請按下方按鈕存入知識庫)", 
        key="text_area_input",
        height=500,
    )

    if st.button("✅ 確認存入正式知識庫 (RAG)"):
        final_text = st.session_state.get('text_area_input', "").strip()
        if final_text:
            with st.spinner("正在同步至正式資料庫..."):
                final_docs = [Document(page_content=final_text, metadata={})]
                # 摘要索引模式：建立包含 FAISS、BM25、原文對照表的字典
                st.session_state['vector_db'] = build_vector_store(final_docs, summarize_model=target_model)
                # 儲存 FAISS 部分到本地端
                st.session_state['vector_db']["vector_db"].save_local("faiss_index_storage")
                st.session_state['manual_context'] = final_text
                st.success(f"✅ 已建立摘要索引 RAG 資料庫 (共 {len(final_text)} 字)")
        
            st.rerun()
        else:
            st.error("內容為空，無法存入")
  
# --- 頁面二：正式知識庫管理 ---
elif page == "正式資料庫管理":
    st.title("正式資料庫管理")
    st.info("這裡顯示的是 AI 目前在對話中使用的最終版本內容，可以在此修正錯誤並重新更新資料庫。")

    if 'manual_context' in st.session_state and st.session_state['manual_context']:
        col1, col2 = st.columns(2)
        current_text = st.session_state['manual_context']
        
        with col1:
            st.metric("📊 目前總字數", f"{len(current_text)} 字")
        with col2:
            db_status = "已啟用 ⚡ (摘要索引 + Rerank)" if 'vector_db' in st.session_state else "純文字模式 📄"
            st.metric("狀態", db_status)

        st.divider()
        st.subheader("📝 修正正式資料內容")
        new_fixed_text = st.text_area(
            "您可以直接在此修改文字，修正後請按下方更新按鈕：",
            value=current_text,
            height=600,
            key="final_db_editor"
        )

        if st.button("🔥 修正並重新上傳至正式資料庫 (更新 RAG)"):
            if new_fixed_text:
                with st.spinner("正在根據修正內容重新建立資料庫..."):
                    st.session_state['manual_context'] = new_fixed_text
                    new_docs = [Document(page_content=new_fixed_text, metadata={"page": "修正版本"})]
                    st.session_state['vector_db'] = build_vector_store(new_docs, summarize_model=target_model)
                    st.success(f"✅ 修正完成，已重新建立摘要索引 RAG 資料庫（共 {len(new_fixed_text)} 字）！")
                    st.balloons()
                    st.rerun()
            else:
                st.error("內容不能為空！")
    else:
        st.warning("⚠️ 目前正式知識庫內沒有資料，請先前往「手冊解析與校對」頁面存入內容。")
    
# --- 頁面三：獨立對話頁面 ---
elif page == "AI對話機器人":
    st.title("AI對話機器人")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("請問關於公司的規定..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):  
            context = ""
            current_k = st.session_state.get('dynamic_k', 5)
            scored_docs = []
            
            # 獲取 RAG 背景資料（摘要索引檢索）
            if st.session_state.get('vector_db'):
                context, scored_docs = get_relevant_context(
                    prompt, 
                    st.session_state['vector_db'], 
                    current_k
                )
                
                # --- Debug 展開視窗 ---
                with st.expander(f"🔍 摘要索引檢索 + Rerank 最終精選原文 (Debug) - 目前 K={current_k}"):
                    if context == "__NO_RELEVANT_CONTEXT__":
                        st.warning("⚠️ 所有候選段落的 Rerank 分數均低於門檻，知識庫中找不到相關內容。")
                    else:
                        for i, (score, d) in enumerate(scored_docs):
                            st.write(f"🥇 名次 {i+1} | 🎯 Rerank 得分: `{score:.4f}`")
                            st.code(d.page_content)
            else:
                context = st.session_state.get('manual_context', "")

            # Score Threshold 攔截：知識庫無相關內容，直接回覆，不進 LLM
            if context == "__NO_RELEVANT_CONTEXT__":
                answer = "📭 手冊中未找到與此問題相關的內容，建議洽詢相關部門或換個關鍵字再試。"
            else:
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

                full_prompt = f"【手冊內容】：\n{context}\n\n當前問題：{prompt}"
                chat_messages = [
                    {'role': 'system', 'content': system_instruction},
                    {'role': 'user', 'content': full_prompt}
                ]

                try:
                    response = chat(
                        model=target_model,
                        messages=chat_messages,
                        options={"temperature": 0.0, "num_predict": 2000}
                    )
                    answer = response.message.content
                except Exception as e:
                    answer = f"連線失敗：{e}"

        st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})