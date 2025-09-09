import os
import uuid
import sqlite3
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

# 初始化应用
app = FastAPI(title="RAG Demo")

# 创建必要的目录
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化向量数据库客户端
chroma_client = chromadb.PersistentClient(path="data/chroma")

# 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# 初始化重排模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 初始化SQLite数据库
def init_db():
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id TEXT PRIMARY KEY,
                  kb_id TEXT,
                  filename TEXT,
                  original_filename TEXT,
                  uploaded_at TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# 添加文档记录
def add_document(kb_id, filename, original_filename):
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    doc_id = str(uuid.uuid4())
    c.execute("INSERT INTO documents (id, kb_id, filename, original_filename, uploaded_at) VALUES (?, ?, ?, ?, ?)",
              (doc_id, kb_id, filename, original_filename, datetime.now()))
    conn.commit()
    conn.close()
    return doc_id

# 获取知识库中的文档
def get_documents(kb_id):
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE kb_id = ? ORDER BY uploaded_at DESC", (kb_id,))
    documents = [dict(row) for row in c.fetchall()]
    conn.close()
    return documents

# 文本分块函数
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 存储文档到向量数据库
def store_document_in_vector_db(kb_id, doc_id, text):
    # 获取或创建集合
    collection_name = f"kb_{kb_id}"
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(collection_name)
    
    # 分块处理文本
    chunks = chunk_text(text)
    
    # 生成嵌入
    embeddings = embedding_model.encode(chunks)
    
    # 准备元数据
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    
    # 添加到集合
    collection.add(
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=chunks,
        ids=ids
    )

# 多路召回检索
def multi_retrieval(query, kb_ids, top_k=5):
    results = []
    
    for kb_id in kb_ids:
        try:
            collection = chroma_client.get_collection(f"kb_{kb_id}")
            
            # 方法1: 基于嵌入相似度
            query_embedding = embedding_model.encode([query]).tolist()
            embedding_results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # 方法2: 基于文本相似度
            text_results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # 合并结果
            combined_docs = []
            seen_ids = set()
            
            # 处理嵌入检索结果
            if embedding_results['documents']:
                for i, doc_list in enumerate(embedding_results['documents']):
                    for j, doc in enumerate(doc_list):
                        doc_id = embedding_results['ids'][i][j]
                        if doc_id not in seen_ids:
                            combined_docs.append({
                                "text": doc,
                                "score": 1 - embedding_results['distances'][i][j],  # 转换为相似度分数
                                "source": f"嵌入相似度 (知识库: {kb_id})"
                            })
                            seen_ids.add(doc_id)
            
            # 处理文本检索结果
            if text_results['documents']:
                for i, doc_list in enumerate(text_results['documents']):
                    for j, doc in enumerate(doc_list):
                        doc_id = text_results['ids'][i][j]
                        if doc_id not in seen_ids:
                            combined_docs.append({
                                "text": doc,
                                "score": text_results['distances'][i][j],  # 这里可能是相似度分数
                                "source": f"文本相似度 (知识库: {kb_id})"
                            })
                            seen_ids.add(doc_id)
            
            results.extend(combined_docs)
        except Exception as e:
            print(f"检索知识库 {kb_id} 时出错: {e}")
            continue
    
    return results

# 重排检索结果
def rerank_results(query, retrieved_docs, top_k=5):
    if not retrieved_docs:
        return []
    
    # 准备用于重排的数据
    pairs = [(query, doc["text"]) for doc in retrieved_docs]
    
    # 使用交叉编码器进行重排
    scores = reranker.predict(pairs)
    
    # 将分数与文档关联
    for i, doc in enumerate(retrieved_docs):
        doc["rerank_score"] = float(scores[i])
    
    # 按重排分数排序
    reranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_docs[:top_k]

# 首页路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 添加文档API
@app.post("/api/documents")
async def add_document_api(kb_id: str = Form(...), file: UploadFile = File(...)):
    # 保存文件
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = f"static/uploads/{filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 添加到数据库
    doc_id = add_document(kb_id, filename, file.filename)
    
    # 读取文件内容 (这里简化处理，实际应该根据文件类型解析)
    try:
        text = content.decode('utf-8')
    except:
        text = str(content)
    
    # 存储到向量数据库
    store_document_in_vector_db(kb_id, doc_id, text)
    
    return JSONResponse(content={"status": "success", "doc_id": doc_id, "kb_id": kb_id})

# 获取文档列表API
@app.get("/api/documents/{kb_id}")
async def get_documents_api(kb_id: str):
    documents = get_documents(kb_id)
    return JSONResponse(content=documents)

# WebSocket聊天连接
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'message':
                message = data['message']
                selected_kbs = data.get('selected_kbs', [])
                
                # 从选定的知识库中检索相关内容
                context = ""
                if selected_kbs:
                    # 多路召回检索
                    retrieved_docs = multi_retrieval(message, selected_kbs, top_k=10)
                    
                    # 重排检索结果
                    reranked_docs = rerank_results(message, retrieved_docs, top_k=5)
                    
                    # 构建上下文
                    context = "\n\n".join([f"[来源: {doc['source']}]\n{doc['text']}" for doc in reranked_docs])
                    
                    await websocket.send_json({
                        "type": "context",
                        "context": f"已从 {len(selected_kbs)} 个知识库中检索到 {len(reranked_docs)} 条相关信息"
                    })
                
                # 模拟LLM生成回复 (实际应用中应调用真实的LLM API)
                response_text = generate_response(message, context)
                
                await websocket.send_json({
                    "type": "response",
                    "message": response_text
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")

# 模拟LLM生成回复
def generate_response(query, context):
    # 这里只是模拟，实际应该调用真实的LLM API
    if context:
        return f"基于您提供的上下文，我找到了以下相关信息：\n\n{context[:1000]}...\n\n根据这些信息，我对您的问题'{query}'的回答是：这是一个很好的问题，相关文档提供了有用的背景信息。"
    else:
        return f"您好！您问的是：'{query}'。这是一个很好的问题，但我没有找到相关的背景信息来帮助回答。请确保您已经上传了相关文档到知识库。"

# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
