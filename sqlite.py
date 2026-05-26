import sqlite3
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = PROJECT_ROOT / "ecommerce_agent"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "bge-large-zh-v1.5"
PRODUCT_VECTOR_PATH = AGENT_ROOT / "product_vector_store"
ORDER_DB_PATH = AGENT_ROOT / "ecommerce_orders.db"


# ======================
# 1. 手动创建商品知识库（15个商品）
# ======================
def create_product_knowledge_base():
    """创建商品知识库并保存到本地"""
    # 15个商品信息
    products = [
        {
            "id": "001",
            "name": "纯棉T恤",
            "description": "100%纯棉材质，透气舒适，适合夏季穿着",
            "specifications": "S/M/L/XL",
            "price": "99元",
            "activity": "满200减30，可叠加使用"
        },
        {
            "id": "002",
            "name": "牛仔裤",
            "description": "修身版型，弹力面料，经典款式",
            "specifications": "28/29/30/31/32（腰围）",
            "price": "159元",
            "activity": "第二件半价"
        },
        {
            "id": "003",
            "name": "运动鞋",
            "description": "轻便透气，缓震鞋底，适合跑步健身",
            "specifications": "39/40/41/42/43/44",
            "price": "299元",
            "activity": "会员专享8折"
        },
        {
            "id": "004",
            "name": "连衣裙",
            "description": "雪纺材质，碎花图案，优雅大方",
            "specifications": "S/M/L",
            "price": "179元",
            "activity": "满300减50"
        },
        {
            "id": "005",
            "name": "夹克外套",
            "description": "防风防水面料，春秋季适用",
            "specifications": "M/L/XL/XXL",
            "price": "259元",
            "activity": "新品上市，暂无活动"
        },
        {
            "id": "006",
            "name": "羊毛衫",
            "description": "含羊毛成分，保暖舒适",
            "specifications": "S/M/L/XL",
            "price": "199元",
            "activity": "满2件减100"
        },
        {
            "id": "007",
            "name": "休闲裤",
            "description": "棉质混纺，宽松版型，日常穿着舒适",
            "specifications": "M/L/XL",
            "price": "129元",
            "activity": "满150减20"
        },
        {
            "id": "008",
            "name": "卫衣",
            "description": "加绒加厚，连帽设计，时尚休闲",
            "specifications": "S/M/L/XL",
            "price": "149元",
            "activity": "限时折扣，直降30元"
        },
        {
            "id": "009",
            "name": "衬衫",
            "description": "免烫处理，商务休闲两用",
            "specifications": "38/39/40/41/42",
            "price": "169元",
            "activity": "满300减60"
        },
        {
            "id": "010",
            "name": "羽绒服",
            "description": "90%白鸭绒填充，轻便保暖",
            "specifications": "M/L/XL/XXL",
            "price": "499元",
            "activity": "预售优惠，定金50抵100"
        },
        {
            "id": "011",
            "name": "帆布鞋",
            "description": "经典款式，舒适百搭，适合日常穿着",
            "specifications": "35/36/37/38/39/40",
            "price": "79元",
            "activity": "买一送一"
        },
        {
            "id": "012",
            "name": "背包",
            "description": "大容量设计，防水面料，适合通勤旅行",
            "specifications": "均码（黑色/灰色/蓝色）",
            "price": "199元",
            "activity": "满200减40"
        },
        {
            "id": "013",
            "name": "帽子",
            "description": "棉质材质，防晒透气，时尚简约",
            "specifications": "均码（可调节）",
            "price": "59元",
            "activity": "3件起9折"
        },
        {
            "id": "014",
            "name": "围巾",
            "description": "羊毛混纺，柔软保暖，多种颜色可选",
            "specifications": "均码（红色/蓝色/灰色/黑色）",
            "price": "89元",
            "activity": "满100减20"
        },
        {
            "id": "015",
            "name": "手套",
            "description": "加绒加厚，触屏设计，冬季必备",
            "specifications": "M/L（黑色/棕色）",
            "price": "69元",
            "activity": "买二送一"
        }
    ]

    # 将商品信息转换为文档格式
    documents = []
    for product in products:
        content = f"商品ID：{product['id']}\n"
        content += f"名称：{product['name']}\n"
        content += f"描述：{product['description']}\n"
        content += f"规格：{product['specifications']}\n"
        content += f"价格：{product['price']}\n"
        content += f"活动：{product['activity']}"

        documents.append(Document(
            page_content=content,
            metadata={"source": f"product_{product['id']}.txt"}
        ))

    # 初始化嵌入模型（使用本地模型路径，避免下载问题）
    local_model_path = str(EMBEDDING_MODEL_PATH)  # 替换为你的本地模型路径
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 创建并保存向量库
    vector_store = FAISS.from_documents(documents, embeddings)
    if not os.path.exists(PRODUCT_VECTOR_PATH):
        vector_store.save_local(str(PRODUCT_VECTOR_PATH))
        print(f"商品知识库创建完成，已保存到 {PRODUCT_VECTOR_PATH} 目录")
    else:
        print("商品知识库已存在，无需重复创建")

    return vector_store


# ======================
# 2. 初始化SQLite订单表（10条测试数据）
# ======================
def init_order_database():
    """初始化订单数据库并插入测试数据"""
    # 连接数据库（如果不存在则创建）
    conn = sqlite3.connect(str(ORDER_DB_PATH))
    cursor = conn.cursor()

    # 创建订单表（确保字段完整且与插入数据匹配）
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS orders
                   (
                       order_id
                       TEXT
                       PRIMARY
                       KEY,
                       user_id
                       TEXT,
                       product_ids
                       TEXT,
                       status
                       TEXT,
                       total_amount
                       REAL,
                       create_time
                       TEXT,
                       pay_time
                       TEXT,
                       ship_time
                       TEXT,
                       receive_time
                       TEXT,
                       logistics_info
                       TEXT
                   )
                   ''')

    # 10条测试订单数据
    test_orders = [
        ("12345", "user001", "001,003", "已签收", 398.0, "2023-10-01 09:30:00", "2023-10-01 10:15:00",
         "2023-10-02 14:20:00", "2023-10-04 16:45:00", "圆通快递: YT1234567890"),
        ("12346", "user002", "002", "已发货", 159.0, "2023-10-02 11:20:00", "2023-10-02 11:30:00",
         "2023-10-03 08:10:00", None, "中通快递: ZT0987654321"),
        ("12347", "user003", "004,006", "已付款", 378.0, "2023-10-02 15:40:00", "2023-10-02 16:05:00", None, None,
         None),
        ("12348", "user004", "005", "待付款", 259.0, "2023-10-03 09:10:00", None, None, None, None),
        ("12349", "user005", "007,008,009", "已签收", 447.0, "2023-10-03 14:30:00", "2023-10-03 15:00:00",
         "2023-10-04 09:20:00", "2023-10-06 11:30:00", "顺丰速运: SF1122334455"),
        ("12350", "user006", "010", "已取消", 499.0, "2023-10-04 10:20:00", None, None, None, None),
        ("12351", "user007", "001,008", "已发货", 248.0, "2023-10-04 16:50:00", "2023-10-04 17:10:00",
         "2023-10-05 10:30:00", None, "韵达快递: YD5566778899"),
        ("12352", "user008", "003,005", "已付款", 558.0, "2023-10-05 08:40:00", "2023-10-05 09:05:00", None, None,
         None),
        ("12353", "user009", "006,007", "已签收", 328.0, "2023-10-05 13:20:00", "2023-10-05 14:00:00",
         "2023-10-06 09:15:00", "2023-10-08 15:20:00", "圆通快递: YT9876543210"),
        ("12354", "user010", "002,009", "待付款", 328.0, "2023-10-06 11:10:00", None, None, None, None)
    ]

    # 插入测试数据
    cursor.executemany('''
                       INSERT
                       OR IGNORE INTO orders 
    (order_id, user_id, product_ids, status, total_amount, create_time, pay_time, ship_time, receive_time, logistics_info)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', test_orders)

    conn.commit()
    conn.close()
    print(f"订单数据库初始化完成，已创建 {ORDER_DB_PATH} 并插入10条测试数据")


# 执行初始化
if __name__ == "__main__":
    create_product_knowledge_base()
    init_order_database()
