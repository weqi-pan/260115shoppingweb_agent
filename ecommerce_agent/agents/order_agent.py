import os
import sqlite3


class OrderAgent:
    def __init__(self, db_path=None):
        # 初始化订单查询工具
        if db_path is None:
            try:
                from ecommerce_agent.config import ORDER_DB_PATH
            except ModuleNotFoundError:
                from config import ORDER_DB_PATH
            db_path = ORDER_DB_PATH
        self.db_path = db_path
        self.order_tool = self._create_order_tool()

    def _create_order_tool(self):
        """创建订单查询工具"""
        from langchain_core.tools import StructuredTool

        def query_order_with_product(order_id: str):
            """查询订单信息（含商品ID）"""
            db_path = self.db_path
            if not os.path.exists(db_path):
                return f"错误：未找到订单数据库文件（{db_path}）"

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                               SELECT status, logistics_info, total_amount, create_time, product_ids, receive_time
                               FROM orders
                               WHERE order_id = ?
                               """, (order_id,))
                result = cursor.fetchone()
                conn.close()

                if not result:
                    return f"未找到订单编号为 {order_id} 的信息"
                status, logistics, amount, create_time, product_ids, receive_time = result
                response = f"订单 {order_id} 信息：\n"
                response += f"- 状态：[{status}]\n"
                response += f"- 总金额：[{amount}元]\n"
                response += f"- 创建时间：[{create_time}]\n"
                response += f"- 签收时间：[{receive_time}]\n"
                response += f"- 商品ID：[{product_ids}]\n"
                if logistics:
                    response += f"- 物流信息：[{logistics}]"
                return response
            except Exception as e:
                return f"查询失败：{str(e)}"

        return StructuredTool.from_function(
            func=query_order_with_product,
            name="query_order",
            description="查询订单详情（含商品ID），参数为order_id（订单编号，如12345）"
        )

    def get_tool(self):
        """提供给接入Agent的工具接口"""
        return self.order_tool
