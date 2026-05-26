# dianshan-agent
用于查询电商信息的agent系统
=
本系统是基于lm studio、langchain、flask、sqlite、html的小型查询电商信息的agent系统
能够查询纯订单问题、纯商品问题、复合问题（如：订单12345的T恤多少钱）

文件清单
-
###
sqlite.py用于建立SQlite数据库和知识向量库的脚本，会生成product_vector_store知识向量库的目录和ecommerce_orders.db订单数据库表
###
bge-large-zh-v1.5目录存放的是中文语义理解模型
###
ecommerce_agent目录下存放有agents目录、执行文件app.py、config.py配置文件、qian.html前端网页
###
agents目录下存放三个agent,分别是access_agent.py,order_agent.py,product_agent.py
###
access_agent.py是接入agent用于理解用户问题和协调，order_agent.py和product_agent.py协调工作
###
order_agent.py是订单agent用于查询SQlite数据库的订单信息
###
product_agent.py是商品agent用于查询知识向量库的商品信息

这里的order_agent.py和product_agent.py不是真正意义上的agent而是封装成的工具，只有access_agent.py是真正接入大模型的agent

注意！！！ 
在config.py要配置相对的bge-large-zh-v1.5目录地址和product_vector_store知识向量库的目录地址
在order_agent.py要配置好对应的数据库： 
```
def query_order_with_product(order_id: str):
                """查询订单信息（含商品ID）"""
                db_path = "ecommerce_orders.db"#数据库地址
                if not os.path.exists(db_path):
                     return f"错误：未找到订单数据库文件（{db_path}）"
```

运行前提
-
需要下载lm studio软件，在本地下载部署大模型，我这里用的是qwen/qwen3-14b模型，在lm studio中只有带锤子的模型才支持调用工具，若想用其他模型请在app.py配置
```
llm_config = {
    "api_key": OPENAI_API_KEY,
    "base_url": BASE_URL,
    "model_name": "qwen/qwen3-14b",#修改成你需要的大模型名称，在lm studio有
    "temperature": 0.0,#这个是设定大模型的生成的随机性
    "max_tokens": 4096#这个是设定大模型的最大tokens
}
```
环境：
```
langchain-community == 0.3.27
langchain-huggingface == 0.3.1
langchain-core == 0.3.72
langchain-openai == 0.3.28
langchain == 0.3.27
Flask == 3.1.0
flask-cors == 6.0.1
```
运行结果
-
先启动app.py再用浏览器打开qian.html,运行结果如下：
<img width="1168" height="857" alt="f6d0a2558b72b5aab0e222ac89dafe70" src="https://github.com/user-attachments/assets/ede76a9a-e62f-4248-9e3e-a78bbcbc8638" />







