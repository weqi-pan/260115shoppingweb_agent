from flask import Flask, request, jsonify
from flask_cors import CORS  # 新增：导入CORS
from config import (
    OPENAI_API_KEY, BASE_URL, EMBEDDING_MODEL_PATH,
    PRODUCT_VECTOR_PATH, ORDER_DB_PATH, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
)
from ecommerce_agent.agents import AccessAgent

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 新增：启用CORS，允许跨域请求

# 初始化接入Agent
llm_config = {
    "api_key": OPENAI_API_KEY,
    "base_url": BASE_URL,
    "model_name": "qwen/qwen3-14b",
    "temperature": 0.0,
    "max_tokens": 4096
}

access_agent = AccessAgent(
    llm_config=llm_config,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    product_vector_path=PRODUCT_VECTOR_PATH,
    order_db_path=ORDER_DB_PATH
)


# API接口
@app.route('/api/query', methods=['POST'])
def query():
    """处理用户查询的API接口"""
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "缺少参数: question"}), 400

    try:
        question = data["question"]
        response = access_agent.handle_question(question)
        return jsonify({
            "success": True,
            "response": response
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
