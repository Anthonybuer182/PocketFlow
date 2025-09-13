#!/usr/bin/env python3
"""
清理ChromaDB数据并重新初始化
"""
import os
import shutil
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def cleanup_chroma():
    print("开始清理ChromaDB数据...")
    
    # 关闭ChromaDB连接
    try:
        chroma_client = chromadb.PersistentClient(path="data/chroma")
        collections = chroma_client.list_collections()
        print(f"找到 {len(collections)} 个集合")
        
        for collection in collections:
            print(f"删除集合: {collection.name}")
            chroma_client.delete_collection(collection.name)
            
    except Exception as e:
        print(f"清理集合时出错: {e}")
    
    # 删除ChromaDB数据目录
    chroma_path = "data/chroma"
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            print(f"已删除目录: {chroma_path}")
        except Exception as e:
            print(f"删除目录时出错: {e}")
    
    # 重新创建目录
    os.makedirs(chroma_path, exist_ok=True)
    print(f"已重新创建目录: {chroma_path}")
    
    # 初始化嵌入模型
    print("初始化嵌入模型...")
    embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    # 自定义嵌入函数
    def custom_embedding_function(texts):
        embeddings = embedding_model.encode(texts)
        return embeddings.tolist()
    
    # 重新初始化ChromaDB客户端
    print("重新初始化ChromaDB客户端...")
    chroma_client = chromadb.PersistentClient(
        path="data/chroma",
        settings=Settings(anonymized_telemetry=False)
    )
    
    print("清理完成！现在可以重新启动应用了。")

if __name__ == "__main__":
    cleanup_chroma()
