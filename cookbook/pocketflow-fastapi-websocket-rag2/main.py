import os
import uuid
import sqlite3
import json
import asyncio
import logging
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
from utils.stream_llm import stream_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 初始化应用
app = FastAPI(title="RAG Demo")
logger.info("FastAPI应用初始化完成")

# 创建必要的目录
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
logger.info("创建必要的目录: static/uploads, data")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger.info("静态文件挂载完成")

# 初始化向量数据库客户端
chroma_client = chromadb.PersistentClient(path="data/chroma")
logger.info("向量数据库客户端初始化完成")

# 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("嵌入模型初始化完成: all-MiniLM-L6-v2")
# 初始化重排模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
logger.info("重排模型初始化完成: cross-encoder/ms-marco-MiniLM-L-6-v2")

# 初始化SQLite数据库
def init_db():
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    # # 删除旧表（如果存在）
    # c.execute('DROP TABLE IF EXISTS documents')
    # 创建新表（如果不存在）
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  original_filename TEXT,
                  uploaded_at TIMESTAMP)''')
    conn.commit()
    conn.close()
    logger.info("SQLite数据库初始化完成")

init_db()

# 添加文档记录
def add_document(filename, original_filename):
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    doc_id = str(uuid.uuid4())
    c.execute("INSERT INTO documents (id, filename, original_filename, uploaded_at) VALUES (?, ?, ?, ?)",
              (doc_id, filename, original_filename, datetime.now()))
    conn.commit()
    conn.close()
    logger.info(f"添加文档记录: ID={doc_id}, 文件名={original_filename}, 存储名={filename}")
    return doc_id

# 获取所有文档
def get_documents():
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    documents = [dict(row) for row in c.fetchall()]
    conn.close()
    return documents

# 获取单个文档信息
def get_document(doc_id):
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    document = dict(result) if result else None
    conn.close()
    return document

# 检查文档是否已存在（通过原始文件名）
def document_exists(original_filename):
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE original_filename = ?", (original_filename,))
    result = c.fetchone()
    document = dict(result) if result else None
    conn.close()
    return document is not None

# 删除文档
def delete_document(doc_id):
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    
    # 获取文件名
    c.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    if result:
        filename = result[0]
        logger.info(f"开始删除文档: ID={doc_id}, 文件名={filename}")
        
        # 删除文件
        file_path = f"static/uploads/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除文件: {file_path}")
        else:
            logger.warning(f"文件不存在: {file_path}")
        
        # 删除向量数据库中的集合
        try:
            collection_name = f"doc_{doc_id}"
            chroma_client.delete_collection(collection_name)
            logger.info(f"已删除向量集合: {collection_name}")
        except Exception as e:
            logger.error(f"删除向量集合时出错: {e}")
        
        # 删除数据库记录
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        logger.info(f"已删除数据库记录: 文档ID={doc_id}")
    
    conn.close()
    logger.info(f"删除操作结果: {'成功' if result is not None else '失败'}")
    return result is not None

# 文本分块函数（递归分块）
def chunk_text(text, chunk_size=500, overlap=0):
    """
    递归分块函数，按分隔符优先级顺序找到一个分隔符进行分块，
    如果块仍然大于chunk_size则递归分块
    
    Args:
        text: 要分块的文本
        chunk_size: 目标块大小
        overlap: 块之间的重叠大小
    
    Returns:
        list: 分块后的文本列表
    """
    # 分隔符优先级列表
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def recursive_chunk(text, separators, chunk_size):
        """递归分块辅助函数"""
        if len(text) <= chunk_size:
            return [text]
        
        # 按顺序查找第一个可用的分隔符
        for sep in separators:
            if sep == "":  # 最后一个分隔符是空字符串，直接按长度分割
                # 直接按chunk_size分割
                chunks = []
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    if end > len(text):
                        end = len(text)
                    chunk = text[start:end]
                    # 如果块仍然过大，递归分块
                    if len(chunk) > chunk_size:
                        sub_chunks = recursive_chunk(chunk, separators, chunk_size)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk)
                    start = end
                return chunks
            
            # 查找分隔符位置
            if sep in text:
                # 找到分隔符，在分隔符处分割
                parts = text.split(sep)
                chunks = []
                current_chunk = ""
                
                for i, part in enumerate(parts):
                    # 计算添加分隔符后的长度（除了第一个部分）
                    add_length = len(sep) if i > 0 and current_chunk else 0
                    
                    # 如果当前块加上新部分不超过chunk_size，则合并
                    potential_length = len(current_chunk) + add_length + len(part)
                    if potential_length <= chunk_size:
                        if current_chunk:
                            current_chunk += sep + part
                        else:
                            current_chunk = part
                    else:
                        # 当前块已满，添加到结果中
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # 如果单个部分就超过chunk_size，需要递归分块
                        if len(part) > chunk_size:
                            sub_chunks = recursive_chunk(part, separators, chunk_size)
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = part
                
                # 添加最后一个块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 检查是否有块仍然过大，需要进一步分块
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > chunk_size:
                        sub_chunks = recursive_chunk(chunk, separators, chunk_size)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks
        
        # 如果没有找到任何分隔符，直接按长度分割
        if len(text) > chunk_size:
            mid = len(text) // 2
            left_chunks = recursive_chunk(text[:mid], separators, chunk_size)
            right_chunks = recursive_chunk(text[mid:], separators, chunk_size)
            return left_chunks + right_chunks
        else:
            return [text]
    
    # 执行递归分块
    chunks = recursive_chunk(text, separators, chunk_size)
    
    # 应用重叠（如果指定了重叠并且有多个块）
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                # 第一个块保持不变
                overlapped_chunks.append(chunks[i])
            else:
                # 从上一个块的末尾取overlap个字符添加到当前块的开头
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
                # 确保重叠后不超过chunk_size
                combined = overlap_text + chunks[i]
                if len(combined) > chunk_size:
                    # 如果超过大小，需要调整
                    excess = len(combined) - chunk_size
                    overlapped_chunks.append(combined[excess:])  # 截断开头部分
                else:
                    overlapped_chunks.append(combined)
        return overlapped_chunks
    
    return chunks

# 存储文档到向量数据库
def store_document_in_vector_db(doc_id, text):
    # 获取或创建集合
    collection_name = f"doc_{doc_id}"
    try:
        collection = chroma_client.get_collection(collection_name)
        logger.info(f"获取现有向量集合: {collection_name}")
    except:
        collection = chroma_client.create_collection(collection_name)
        logger.info(f"创建新的向量集合: {collection_name}")
    
    # 分块处理文本
    chunks = chunk_text(text)
    logger.info(f"文档分块完成: 文档ID={doc_id}, 块数={len(chunks)}")
    
    # 生成嵌入
    embeddings = embedding_model.encode(chunks)
    logger.info(f"嵌入生成完成: 文档ID={doc_id}, 嵌入维度={embeddings.shape}")
    
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
    logger.info(f"文档存储到向量数据库完成: 文档ID={doc_id}, 总块数={len(chunks)}")

# 多路召回检索
def multi_retrieval(query, doc_ids, top_k=5):
    results = []
    
    for doc_id in doc_ids:
        try:
            collection = chroma_client.get_collection(f"doc_{doc_id}")
            
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
                        chunk_id = embedding_results['ids'][i][j]
                        if chunk_id not in seen_ids:
                            combined_docs.append({
                                "text": doc,
                                "score": 1 - embedding_results['distances'][i][j],  # 转换为相似度分数
                                "source": f"嵌入相似度 (文档: {doc_id})"
                            })
                            seen_ids.add(chunk_id)
            
            # 处理文本检索结果
            if text_results['documents']:
                for i, doc_list in enumerate(text_results['documents']):
                    for j, doc in enumerate(doc_list):
                        chunk_id = text_results['ids'][i][j]
                        if chunk_id not in seen_ids:
                            combined_docs.append({
                                "text": doc,
                                "score": text_results['distances'][i][j],  # 这里可能是相似度分数
                                "source": f"文本相似度 (文档: {doc_id})"
                            })
                            seen_ids.add(chunk_id)
            
            results.extend(combined_docs)
        except Exception as e:
            logger.error(f"检索文档 {doc_id} 时出错: {e}")
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
async def add_document_api(file: UploadFile = File(...)):
    logger.info(f"开始处理文件上传: 文件名={file.filename}, 文件大小={file.size}")
    
    # 检查文档是否已存在
    if document_exists(file.filename):
        logger.warning(f"文档已存在，拒绝重复上传: {file.filename}")
        return JSONResponse(
            content={
                "status": "error", 
                "message": f"文档 '{file.filename}' 已上传过，不能重复上传"
            },
            status_code=400
        )
    
    # 保存文件
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = f"static/uploads/{filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"文件保存成功: 存储路径={file_path}")
    
    # 添加到数据库
    doc_id = add_document(filename, file.filename)
    
    # 读取文件内容 (这里简化处理，实际应该根据文件类型解析)
    try:
        text = content.decode('utf-8')
        logger.info(f"文件内容解码成功: 文档ID={doc_id}, 文本长度={len(text)}")
    except:
        text = str(content)
        logger.warning(f"文件内容无法解码为UTF-8, 使用字符串表示: 文档ID={doc_id}")
    
    # 存储到向量数据库
    store_document_in_vector_db(doc_id, text)
    logger.info(f"文档处理完成: 文档ID={doc_id}, 原始文件名={file.filename}")
    
    return JSONResponse(content={"status": "success", "doc_id": doc_id})

# 获取文档列表API
@app.get("/api/documents")
async def get_documents_api():
    documents = get_documents()
    return JSONResponse(content=documents)

# 下载文档API
@app.get("/api/documents/{doc_id}/download")
async def download_document(doc_id: str):
    document = get_document(doc_id)
    if not document:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)
    
    file_path = f"static/uploads/{document['filename']}"
    if not os.path.exists(file_path):
        return JSONResponse(content={"status": "error", "message": "文件不存在"}, status_code=404)
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=document['original_filename'],
        media_type='application/octet-stream'
    )

# 删除文档API
@app.delete("/api/documents/{doc_id}")
async def delete_document_api(doc_id: str):
    success = delete_document(doc_id)
    if success:
        return JSONResponse(content={"status": "success", "message": "文档删除成功"})
    else:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)

# 搜索文档内容API
@app.post("/api/documents/{doc_id}/search")
async def search_document_api(doc_id: str, request: Request):
    # 检查文档是否存在
    document = get_document(doc_id)
    if not document:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)
    
    # 解析请求体
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(content={"status": "error", "message": "搜索关键词不能为空"}, status_code=400)
    except:
        return JSONResponse(content={"status": "error", "message": "无效的请求格式"}, status_code=400)
    
    try:
        # 获取文档对应的向量集合
        collection_name = f"doc_{doc_id}"
        collection = chroma_client.get_collection(collection_name)
        
        # 使用嵌入模型进行语义搜索
        query_embedding = embedding_model.encode([query]).tolist()
        
        # 查询向量数据库
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=20,  # 获取更多结果用于重排
            include=["documents", "distances", "metadatas"]
        )
        
        # 处理搜索结果
        search_results = []
        if results and results['documents']:
            for i, doc_list in enumerate(results['documents']):
                for j, document_text in enumerate(doc_list):
                    # 计算相似度分数（将距离转换为相似度，0-1范围）
                    distance = results['distances'][i][j]
                    similarity_score = 1 - distance  # 距离越小，相似度越高
                    
                    # 获取元数据
                    metadata = results['metadatas'][i][j] if results['metadatas'] else {}
                    page = metadata.get("page", 1)
                    
                    search_results.append({
                        "content": document_text,
                        "score": similarity_score,
                        "page": page,
                        "metadata": metadata
                    })
        
        # 按分数排序（从高到低）
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
        return JSONResponse(content={
            "status": "success",
            "data": search_results,
            "document_name": document['original_filename']
        })
        
    except Exception as e:
        logger.error(f"搜索文档时出错: {e}")
        return JSONResponse(content={
            "status": "error", 
            "message": f"搜索失败: {str(e)}"
        }, status_code=500)

# WebSocket聊天连接
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"收到WebSocket消息: {data}")
            
            if data['type'] == 'message':
                message = data['message']
                selected_docs = data.get('selected_docs', [])
                logger.info(f"处理用户消息: 消息长度={len(message)}, 选择文档数={len(selected_docs)}")
                
                # 从选定的文档中检索相关内容
                context = ""
                if selected_docs:
                    # 多路召回检索
                    retrieved_docs = multi_retrieval(message, selected_docs, top_k=10)
                    logger.info(f"多路召回检索完成: 检索到 {len(retrieved_docs)} 条结果")
                    
                    # 重排检索结果
                    reranked_docs = rerank_results(message, retrieved_docs, top_k=5)
                    logger.info(f"重排完成: 保留 {len(reranked_docs)} 条最相关结果")
                    
                    # 构建上下文
                    context = "\n\n".join([f"[来源: {doc['source']}]\n{doc['text']}" for doc in reranked_docs])
                    
                    await websocket.send_json({
                        "type": "context",
                        "context": f"已从 {len(selected_docs)} 个文档中检索到 {len(reranked_docs)} 条相关信息"
                    })
                
                # 使用OpenAI API流式生成回复
                messages = []
                if context:
                    messages.append({"role": "system", "content": f"基于以下上下文信息回答问题：\n\n{context}\n\n请根据上下文提供准确、相关的回答。"})
                messages.append({"role": "user", "content": message})
                
                # 发送开始响应消息
                await websocket.send_json({
                    "type": "response_start"
                })
                logger.info("开始生成AI回复")
                
                # 流式返回响应
                full_response = ""
                async for chunk in stream_llm(messages):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "response_chunk",
                        "chunk": chunk
                    })
                
                # 发送结束响应消息
                await websocket.send_json({
                    "type": "response_end",
                    "full_response": full_response
                })
                logger.info(f"AI回复生成完成: 回复长度={len(full_response)}")
                
    except WebSocketDisconnect:
        logger.info("客户端断开WebSocket连接")
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
        await websocket.close(code=1011)

# 启动应用
if __name__ == "__main__":
    import uvicorn
    logger.info("启动FastAPI应用服务器...")
    logger.info(f"服务器地址: http://0.0.0.0:8000")
    logger.info(f"API文档地址: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
