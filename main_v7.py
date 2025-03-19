from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Union
import torch
import numpy as np
import logging
import os
from flask_cors import CORS

# 初始化配置
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# GPU 自动检测与初始化
use_cuda = torch.cuda.is_available()
device_count = torch.cuda.device_count() if use_cuda else 0
device = torch.device("cuda" if use_cuda else "cpu")
convert_to_tensor = (device == "cuda")
app.logger.info(f"Detected {device_count} GPUs, using device: {device}")

# 全局模型池
_loaded_models: Dict[str, Union[SentenceTransformer, CrossEncoder]] = {}

MODEL_CONFIG = {
    "inspur-bge-large-zh-v1.5": {
        "type": "embedding",
        "model": "/app/bge-large-zh-v1.5"
    },
    "inspur-bge-reranker-large": {
        "type": "rerank",
        "model": "/app/bge-reranker-large"
    }
}

def validate_embeddings_input(data: dict) -> Union[tuple, None]:
    """OpenAI Embedding 接口参数校验"""
    required_fields = ["input"]
    for field in required_fields:
        if field not in data:
            return (
                jsonify({
                    "error": {
                        "message": f"Missing required field '{field}'",
                        "type": "invalid_request_error",
                        "code": "missing_parameter"
                    }
                }), 
                400
            )
    
    if not isinstance(data["input"], (str, list)):
        return (
            jsonify({
                "error": {
                    "message": "Input must be a string or array of strings",
                    "type": "invalid_request_error",
                    "code": "invalid_input_type"
                }
            }),
            400
        )
    
    if isinstance(data["input"], list) and not all(isinstance(i, str) for i in data["input"]):
        return (
            jsonify({
                "error": {
                    "message": "All elements in input array must be strings",
                    "type": "invalid_request_error",
                    "code": "invalid_array_element"
                }
            }),
            400
        )
    
    return None

def validate_rerank_input(data: dict) -> Union[tuple, None]:
    """OpenAI Rerank 接口参数校验"""
    required_fields = ["query", "documents"]
    for field in required_fields:
        if field not in data:
            return (
                jsonify({
                    "error": {
                        "message": f"Missing required field '{field}'",
                        "type": "invalid_request_error",
                        "code": "missing_parameter"
                    }
                }), 
                400
            )
    
    if not isinstance(data["documents"], list) or len(data["documents"]) == 0:
        return (
            jsonify({
                "error": {
                    "message": "Documents must be a non-empty array",
                    "type": "invalid_request_error",
                    "code": "invalid_documents_format"
                }
            }),
            400
        )
    
    if not all(isinstance(doc, str) for doc in data["documents"]):
        return (
            jsonify({
                "error": {
                    "message": "All documents must be strings",
                    "type": "invalid_request_error",
                    "code": "invalid_document_type"
                }
            }),
            400
        )
    
    return None

def load_model(model_name: str) -> Union[SentenceTransformer, CrossEncoder]:
    """统一模型加载逻辑"""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not supported")
    
    config = MODEL_CONFIG[model_name]
    model_path = config["model"]
    
    if model_path not in _loaded_models:
        app.logger.info(f"Loading {config['type']} model: {model_path}")
        if config["type"] == "embedding":
            model = SentenceTransformer(model_path, device=device)
            if use_cuda:
                 try:
                   model = model.half()  # 尝试半精度转换
                 except RuntimeError:
                   app.logger.warning("半精度转换失败，回退到FP32")
                   model = model.float()
        elif config["type"] == "rerank":
            model = CrossEncoder(model_path, device=device)
        
        _loaded_models[model_path] = model
    
    return _loaded_models[model_path]

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """严格兼容OpenAI的Embedding接口"""
    try:
        data = request.get_json()
        
        # 参数校验
        validation_error = validate_embeddings_input(data)
        if validation_error:
            return validation_error
        
        model_name = data.get("model", "inspur-bge-large-zh-v1.5")
        input_data = data["input"]
        texts = [input_data] if isinstance(input_data, str) else input_data
        
        # 加载模型
        model = load_model(model_name)
        
        # 生成嵌入
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_tensor=convert_to_tensor, 
                device=device
            ).tolist()
        
        # 构造OpenAI响应
        return jsonify({
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
                "prompt_tokens": 0,  # 实际部署需实现token计算
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        app.logger.error(f"Embedding Error: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "server_error"
            }
        }), 500

@app.route('/v1/rerank', methods=['POST'])
def rerank():
    """严格兼容OpenAI的Rerank接口"""
    try:
        data = request.get_json()
        
        # 参数校验
        validation_error = validate_rerank_input(data)
        if validation_error:
            return validation_error
        
        model_name = data.get("model", "inspur-bge-reranker-large")
        query = data["query"]
        documents = data["documents"]
        top_n = data.get("top_n", len(documents))
        
        # 加载模型
        model = load_model(model_name)
        
        # 计算相关性得分
        model_inputs = [[query, doc] for doc in documents]
        scores = model.predict(model_inputs)
        
        # 排序处理
        ranked_indices = np.argsort(scores)[::-1]
        results = [
            {
                "index": int(idx),
                "document": documents[idx],
                "relevance_score": float(scores[idx])
            } for idx in ranked_indices[:top_n]
        ]
        
        # 构造OpenAI响应
        return jsonify({
            "object": "list",
            "results": results,
            "model": model_name,
            "usage": {
                "prompt_tokens": 0,  # 实际部署需实现token计算
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=35000)