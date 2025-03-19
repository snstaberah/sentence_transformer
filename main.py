from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import Union, List, Dict
import numpy as np
import logging
import time
import os
from flask_cors import CORS

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)
current_pid = os.getpid()
# 模型配置（扩展支持 rerank 模型）
MODEL_CONFIG = {
    # Embedding 模型
    "inspur-bge-large-zh-v1.5": {
        "type": "embedding",
        "model": "/app/bge-large-zh-v1.5",
        "tokenizer": "bert-base-uncased"
    },
    
    # Rerank 模型（使用 cross-encoder）
    "inspur-bge-reranker-large": {
        "type": "rerank",
        "model": "/app/bge-reranker-large",
        "tokenizer": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
}

# 全局缓存
_loaded_models: Dict[str, Union[SentenceTransformer, CrossEncoder]] = {}
# _loaded_tokenizers: Dict[str, AutoTokenizer] = {}

def load_components(model_name: str) -> tuple:
    """动态加载模型和分词器（支持 embedding/rerank 类型）"""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not supported")

    config = MODEL_CONFIG[model_name]
    model_id = config["model"]
    # tokenizer_id = config["tokenizer"]

    # 加载分词器
    # if tokenizer_id not in _loaded_tokenizers:
    #     app.logger.info(f"Loading tokenizer: {tokenizer_id}")
    #     _loaded_tokenizers[tokenizer_id] = AutoTokenizer.from_pretrained(tokenizer_id)

    # 加载模型（区分类型）
    if model_id not in _loaded_models:
        app.logger.info(f"Loading model: {model_id}")
        if config["type"] == "embedding":
            _loaded_models[model_id] = SentenceTransformer(model_id)
        elif config["type"] == "rerank":
            _loaded_models[model_id] = CrossEncoder(model_id)

    return _loaded_models[model_id]

def count_rerank_tokens(query: str, documents: List[str], tokenizer: AutoTokenizer) -> int:
    """计算 Rerank 请求的 Token 总数"""
    query_tokens = len(tokenizer.encode(query))
    doc_tokens = sum(len(tokenizer.encode(doc)) for doc in documents)
    return query_tokens + doc_tokens

@app.route('/v1/rerank', methods=['POST'])
def rerank():
    """实现 OpenAI 兼容的 Rerank 接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be JSON"}), 400

        # 解析参数
        query = data.get("query")
        documents = data.get("documents")
        model_name = data.get("model", "text-rerank-001")
        top_n = data.get("top_n", len(documents))  # 默认返回全部

        # 校验输入
        if not query or not documents:
            return jsonify({
                "error": {
                    "message": "Missing required parameters 'query' or 'documents'",
                    "type": "invalid_request_error",
                    "code": "missing_parameter"
                }
            }), 400

        if not isinstance(documents, list) or len(documents) == 0:
            return jsonify({
                "error": {
                    "message": "Documents must be a non-empty array",
                    "type": "invalid_request_error",
                    "code": "invalid_input"
                }
            }), 400

        # 加载模型和分词器
        model = load_components(model_name)
        if MODEL_CONFIG[model_name]["type"] != "rerank":
            raise ValueError(f"Model {model_name} is not a rerank model")

        # 计算相关性得分
        model_inputs = [[query, doc] for doc in documents]
        scores = model.predict(model_inputs)

        # 构造排序结果
        ranked_indices = np.argsort(scores)[::-1]  # 降序排列
        ranked_docs = [{
            "document": documents[idx],
            "index": int(idx),
            "relevance_score": float(scores[idx])
        } for idx in ranked_indices]

        # 应用 top_n 截断
        ranked_docs = ranked_docs[:top_n]

        # 计算 Token 使用量
        # total_tokens = count_rerank_tokens(query, documents, tokenizer)

        # 构造 OpenAI 兼容响应
        return jsonify({
            "object": "list",
            "results": ranked_docs,
            "model": model_name,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        })

    except Exception as e:
        app.logger.error(f"Rerank Error: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "server_error"
            }
        }), 500

# 保留原有的 /v1/embeddings 路由...
@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """严格模拟 OpenAI 的 Embedding 接口"""
    try:
        # 解析请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be JSON"}), 400

        input_data = data.get("input")
        model_name = data.get("model", "text-embedding-ada-002")

        # 校验输入
        if not input_data:
            return jsonify({
                "error": {
                    "message": "Missing required parameter 'input'",
                    "type": "invalid_request_error",
                    "code": "missing_parameter"
                }
            }), 400

        if not isinstance(input_data, (str, list)):
            return jsonify({
                "error": {
                    "message": "Input must be a string or array of strings",
                    "type": "invalid_request_error",
                    "code": "invalid_input"
                }
            }), 400

        # 转换为列表格式
        texts = [input_data] if isinstance(input_data, str) else input_data

        # 加载模型和分词器
        model = load_components(model_name)

        # 计算 Token 数量
        # prompt_tokens = count_tokens(texts, tokenizer)
        prompt_tokens = 0
        start_time = time.perf_counter()
        #app.logger.info(f"encode start: {start_time}")
        # 生成嵌入
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        end_time = time.perf_counter()
        app.logger.info(f"pid: {current_pid} encode cost: {end_time-start_time}")
        # 构造 OpenAI 格式响应
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": idx
                } for idx, embedding in enumerate(embeddings)
            ],
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens  # 简化处理：total_tokens = prompt_tokens
            }
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "server_error"
            }
        }), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=35000)