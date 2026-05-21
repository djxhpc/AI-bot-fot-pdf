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
    
    menu_options = ["手冊解析與校對", "正式資料庫管理", "AI對話機器人", "批次測試"]
    page_selection = st.radio("選單", options=menu_options, index=0)
    
    page = page_selection

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
    retrieval_k = st.slider("最終提供給 AI 的片段數量 (K)", min_value=1, max_value=30, value=15, help="混合搜尋後經由 Rerank 篩選出的最終菁英片段數量。")
    st.session_state['dynamic_k'] = retrieval_k
    score_threshold = st.slider("Rerank 分數門檻", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="低於此分數的段落視為無相關內容")
    st.session_state['score_threshold'] = score_threshold

    st.header("🧹 系統維護")
    if st.button("🗑️ 清除快取紀錄"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        import shutil, os
        if os.path.exists("faiss_index_storage"):
            shutil.rmtree("faiss_index_storage")
        st.toast("快取與知識庫已完全清除！")
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
    from langchain_huggingface import HuggingFaceEmbeddings
    model_name = "BAAI/bge-m3"
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})

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

def _rewrite_query(query: str, history: list, model_name: str) -> str:
    """把短問句結合對話歷史改寫成完整查詢，避免 RAG 搜尋時丟失上下文。"""
    if not history:
        return query
    recent = history[-4:]  # 最近 2 輪
    history_text = "\n".join(
        f"{'用戶' if m['role'] == 'user' else '助手'}：{m['content'][:300]}"
        for m in recent
    )
    rewrite_prompt = (
        "根據以下對話歷史，把「當前問題」改寫成一個完整、不依賴對話歷史也能獨立理解的搜尋查詢。\n"
        "只輸出改寫後的查詢，不要解釋，不要加任何前綴。\n\n"
        f"對話歷史：\n{history_text}\n\n"
        f"當前問題：{query}\n\n"
        "改寫後的查詢："
    )
    try:
        resp = chat(
            model=model_name,
            messages=[{"role": "user", "content": rewrite_prompt}],
            options={"temperature": 0.0, "num_predict": 100},
        )
        return resp.message.content.strip() or query
    except Exception:
        return query

def build_vector_store(docs, summarize_model: str = ""):
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    import uuid, pickle, os

    global_meta_info = _detect_doc_meta(docs)
    sections = _split_into_sections(docs)

    for sec in sections:
        sec.metadata["doc_context"] = global_meta_info if global_meta_info else "通用文本"
        sec.metadata["doc_id"] = str(uuid.uuid4())

    # 加入一個專屬「文件資訊」chunk，讓「哪間公司」等 meta 查詢可以被搜尋到
    if global_meta_info:
        meta_id = str(uuid.uuid4())
        meta_doc = Document(
            page_content=f"本文件基本資訊：{global_meta_info}。本手冊為公司工作規則。",
            metadata={"page": 0, "doc_id": meta_id, "doc_context": global_meta_info},
        )
        index_sections = [meta_doc] + sections
    else:
        index_sections = sections

    id_to_full = {sec.metadata["doc_id"]: sec for sec in index_sections}

    with st.spinner(f"正在建立向量庫（共 {len(sections)} 個段落）..."):
        vector_db = FAISS.from_documents(index_sections, get_embedding_model())
        try:
            import jieba
            bm25_retriever = BM25Retriever.from_documents(
                index_sections,
                preprocess_func=lambda text: list(jieba.cut(text)),
            )
        except ImportError:
            bm25_retriever = BM25Retriever.from_documents(index_sections)

        os.makedirs("faiss_index_storage", exist_ok=True)
        vector_db.save_local("faiss_index_storage")
        with open("faiss_index_storage/sections.pkl", "wb") as f:
            pickle.dump(index_sections, f)
        with open("faiss_index_storage/id_to_full.pkl", "wb") as f:
            pickle.dump(id_to_full, f)

    st.success(f"✅ 建庫完成！共 {len(sections)} 個段落。")

    return {
        "vector_db":      vector_db,
        "bm25_retriever": bm25_retriever,
        "id_to_full":     id_to_full,
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

    SCORE_THRESHOLD = st.session_state.get('score_threshold', 0.3)

    candidate_k = max(k_value * 3, 20)

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
    all_top_k = scored_docs[:k_value]
    top_k_scored = [(score, doc) for score, doc in all_top_k if score >= SCORE_THRESHOLD]

    # 供 Debug 視窗使用：保存全部排序結果（含被門檻過濾的）
    st.session_state['_debug_all_scored'] = all_top_k
    st.session_state['_debug_threshold']  = SCORE_THRESHOLD
    st.session_state['_debug_candidates'] = len(full_text_candidates)

    if not top_k_scored:
        return "__NO_RELEVANT_CONTEXT__", []

    st.sidebar.caption(
        f"⚡ 檢索完成！初篩 {len(full_text_candidates)} 段 ➡️ 門檻通過 {len(top_k_scored)} 段"
    )

    final_docs   = [doc for _, doc in top_k_scored]
    context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
    return context_text, top_k_scored


def get_rag_answer(question: str, model_name: str, k: int) -> tuple[str, str]:
    """批次測試用：非串流版 RAG 問答，回傳 (answer, context_preview)。"""
    hybrid_db = st.session_state.get('vector_db')
    if not hybrid_db:
        return "⚠️ 尚未建立知識庫", ""

    context, scored_docs = get_relevant_context(question, hybrid_db, k)
    if context == "__NO_RELEVANT_CONTEXT__":
        return "📭 手冊中未找到相關內容", ""

    context_preview = context[:300] + ("..." if len(context) > 300 else "")

    system_instruction = (
        "你現在是一位專業的企業行政助手。\n"
        "你的唯一任務是根據使用者提供的【手冊內容】回答問題。\n\n"
        "【強制規則】：\n"
        "1. 只能從下方【手冊內容】中尋找答案，絕對不可使用手冊以外的任何知識。\n"
        "2. 必須完全使用繁體中文（台灣習慣用語）回答。\n"
        "3. 手冊中若找不到答案，請明確說「手冊中未提及此項目」，不要推測或編造。\n"
        "4. 回答結尾必須附上【原文依據】，直接引用手冊中的相關句子。\n"
        "5. 保持專業、客觀、準確。"
    )
    full_prompt = f"【手冊內容】：\n{context}\n\n當前問題：{question}"
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt},
            ],
            options={"temperature": 0.0, "num_predict": 2000},
        )
        return resp.message.content.strip(), context_preview
    except Exception as e:
        return f"連線失敗：{e}", context_preview


# --- 4b. 啟動時從磁碟還原知識庫（須在函數定義後執行）---
if 'vector_db' not in st.session_state:
    import os, pickle
    if os.path.exists("faiss_index_storage") and os.path.exists("faiss_index_storage/id_to_full.pkl"):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.retrievers import BM25Retriever
            _vdb = FAISS.load_local("faiss_index_storage", get_embedding_model(), allow_dangerous_deserialization=True)
            _sections_path = "faiss_index_storage/sections.pkl"
            _legacy_path   = "faiss_index_storage/summary_docs.pkl"
            _src_path = _sections_path if os.path.exists(_sections_path) else _legacy_path
            with open(_src_path, "rb") as f:
                _sections = pickle.load(f)
            with open("faiss_index_storage/id_to_full.pkl", "rb") as f:
                _id_to_full = pickle.load(f)
            try:
                import jieba
                _bm25 = BM25Retriever.from_documents(
                    _sections,
                    preprocess_func=lambda text: list(jieba.cut(text)),
                )
            except ImportError:
                _bm25 = BM25Retriever.from_documents(_sections)
            st.session_state['vector_db'] = {
                "vector_db": _vdb,
                "bm25_retriever": _bm25,
                "id_to_full": _id_to_full,
            }
            ctx_path = "faiss_index_storage/manual_context.txt"
            if os.path.exists(ctx_path):
                with open(ctx_path, "r", encoding="utf-8") as f:
                    st.session_state['manual_context'] = f.read()
        except Exception:
            pass


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
                            documents.append(Document(page_content=clean_text, metadata={"page": i+1}))
                    st.session_state['temp_docs'] = documents 
                    st.session_state['text_area_input'] = "\n\n".join([doc.page_content for doc in documents])
                    st.success(f"解析成功！共 {len(reader.pages)} 頁。")

    with tab_testset:
        st.subheader("MediaTek-Research TCEval-v2")
        df_test = get_test_dataset()
        if "test_notes" not in st.session_state:
            st.session_state.test_notes = df_test.assign(備註="")

        st.data_editor(
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
                # PDF 來源優先用 temp_docs（保留頁碼）；其他來源用純文字
                final_docs = st.session_state.get('temp_docs') or [Document(page_content=final_text, metadata={})]
                # 摘要索引模式：建立包含 FAISS、BM25、原文對照表的字典
                st.session_state['vector_db'] = build_vector_store(final_docs, summarize_model=target_model)
                st.session_state['manual_context'] = final_text
                import os
                os.makedirs("faiss_index_storage", exist_ok=True)
                with open("faiss_index_storage/manual_context.txt", "w", encoding="utf-8") as f:
                    f.write(final_text)
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
                    import os
                    os.makedirs("faiss_index_storage", exist_ok=True)
                    with open("faiss_index_storage/manual_context.txt", "w", encoding="utf-8") as f:
                        f.write(new_fixed_text)
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
            history_so_far = st.session_state.messages[:-1]  # 不含當前這輪

            # 獲取 RAG 背景資料（摘要索引檢索）
            if st.session_state.get('vector_db'):
                # Query Rewriting：把短問句結合歷史改寫成完整查詢，解決上下文斷鏈問題
                search_query = _rewrite_query(prompt, history_so_far, target_model)
                if search_query != prompt:
                    st.caption(f"🔄 查詢改寫：{search_query}")

                context, scored_docs = get_relevant_context(
                    search_query,
                    st.session_state['vector_db'],
                    current_k
                )

                # --- Debug 展開視窗 ---
                all_scored = st.session_state.get('_debug_all_scored', [])
                threshold  = st.session_state.get('_debug_threshold', 0.3)
                n_cands    = st.session_state.get('_debug_candidates', 0)
                with st.expander(f"🔍 Rerank 全部排名（候選 {n_cands} 段，門檻 {threshold:.2f}）- K={current_k}"):
                    if not all_scored:
                        st.warning("⚠️ 無候選段落。")
                    else:
                        for i, (score, d) in enumerate(all_scored):
                            page   = d.metadata.get('page', '?')
                            passed = score >= threshold
                            icon   = "✅" if passed else "❌"
                            st.write(f"{icon} 名次 {i+1} | 分數 `{score:.4f}` | 📄 第 {page} 頁")
                            st.code(d.page_content[:300])

            # 無 RAG 庫：拒絕回答，避免 LLM 憑空捏造
            if not st.session_state.get('vector_db'):
                answer = "⚠️ 尚未建立知識庫，請先至「手冊解析與校對」頁面上傳文件並存入知識庫。"
                st.markdown(answer, unsafe_allow_html=True)
            # Rerank 門檻攔截：知識庫無相關內容
            elif context == "__NO_RELEVANT_CONTEXT__":
                answer = "📭 手冊中未找到與此問題相關的內容，建議洽詢相關部門或換個關鍵字再試。"
                st.markdown(answer, unsafe_allow_html=True)
            else:
                system_instruction = (
                    "你現在是一位專業的企業行政助手。\n"
                    "你的唯一任務是根據使用者提供的【手冊內容】回答問題。\n\n"
                    "【強制規則】：\n"
                    "1. 只能從下方【手冊內容】中尋找答案，絕對不可使用手冊以外的任何知識。\n"
                    "2. 必須完全使用繁體中文（台灣習慣用語）回答。\n"
                    "3. 手冊中若找不到答案，請明確說「手冊中未提及此項目」，不要推測或編造。\n"
                    "4. 回答結尾必須附上【原文依據】，直接引用手冊中的相關句子。\n"
                    "5. 保持專業、客觀、準確。\n"
                    "6. 遇到打招呼，請親切回應，無需引用手冊。\n"
                    "7. 遇到無意義亂碼，請禮貌說明並引導詢問手冊相關問題。"
                )

                full_prompt = f"【手冊內容】：\n{context}\n\n當前問題：{prompt}"
                # 歷史僅保留最近 3 輪（6 條），避免幻覺累積與 context 超長
                chat_messages = [{'role': 'system', 'content': system_instruction}]
                for msg in history_so_far[-6:]:
                    chat_messages.append({'role': msg['role'], 'content': msg['content']})
                chat_messages.append({'role': 'user', 'content': full_prompt})

                placeholder = st.empty()
                answer = ""
                try:
                    for chunk in chat(
                        model=target_model,
                        messages=chat_messages,
                        stream=True,
                        options={"temperature": 0.0, "num_predict": 2000},
                    ):
                        answer += chunk.message.content
                        placeholder.markdown(answer)
                except Exception as e:
                    answer = f"連線失敗：{e}"
                    placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- 頁面四：批次測試 ---
elif page == "批次測試":
    st.title("🧪 批次測試")

    try:
        from test_data import HALLUCINATION_TESTS, RAG_TESTS
    except ImportError:
        st.error("❌ 找不到 test_data.py，請確認該檔案與 new67.py 在同一目錄。")
        st.stop()

    if not st.session_state.get('vector_db'):
        st.warning("⚠️ 尚未建立知識庫，請先至「手冊解析與校對」頁面上傳文件並存入知識庫。")
        st.stop()

    test_set_choice = st.radio(
        "選擇測試集",
        ["幻覺測試 (10題)", "RAG 完整測試 (30題)", "全部 (40題)"],
        horizontal=True,
    )

    if test_set_choice == "幻覺測試 (10題)":
        selected_tests = HALLUCINATION_TESTS
    elif test_set_choice == "RAG 完整測試 (30題)":
        selected_tests = RAG_TESTS
    else:
        selected_tests = HALLUCINATION_TESTS + RAG_TESTS

    st.info(f"共 **{len(selected_tests)}** 題，使用模型：`{target_model}`，K={retrieval_k}")

    with st.expander("📋 預覽題目列表", expanded=False):
        preview_rows = [
            {"題號": t["題號"], "問題": t["問題"], "預期答案": t["預期答案"]}
            for t in selected_tests
        ]
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    if st.button("🚀 開始批次測試", type="primary"):
        results = []
        progress_bar = st.progress(0, text="準備開始...")
        status_text  = st.empty()

        for idx, test in enumerate(selected_tests):
            question = test["問題"]
            expected = test["預期答案"]
            status_text.text(f"正在測試第 {idx + 1}/{len(selected_tests)} 題：{question[:40]}...")

            answer, ctx_preview = get_rag_answer(question, target_model, retrieval_k)

            results.append({
                "題號":   test["題號"],
                "問題":   question,
                "系統回答": answer,
                "預期答案": expected,
                "檢索預覽": ctx_preview,
            })
            progress_bar.progress((idx + 1) / len(selected_tests), text=f"{idx + 1}/{len(selected_tests)} 完成")

        status_text.success(f"✅ 全部 {len(results)} 題測試完成！")
        st.session_state['_batch_results'] = results

    if st.session_state.get('_batch_results'):
        results = st.session_state['_batch_results']
        st.divider()
        st.subheader(f"📊 測試結果（共 {len(results)} 題）")

        for r in results:
            with st.expander(f"題號 {r['題號']}：{r['問題'][:50]}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**🤖 系統回答**")
                    st.write(r["系統回答"])
                with col_b:
                    st.markdown("**✅ 預期答案**")
                    st.write(r["預期答案"])
                if r["檢索預覽"]:
                    st.caption(f"檢索片段預覽：{r['檢索預覽']}")

        df_results = pd.DataFrame([
            {"題號": r["題號"], "問題": r["問題"], "系統回答": r["系統回答"], "預期答案": r["預期答案"]}
            for r in results
        ])
        csv_data = df_results.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ 下載結果 CSV",
            data=csv_data,
            file_name="batch_test_results.csv",
            mime="text/csv",
        )