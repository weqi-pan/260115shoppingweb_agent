import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import StructuredTool


class ProductAgent:
    def __init__(self, embedding_model_path, product_vector_path):
        self.embedding_model_path = embedding_model_path
        self.product_vector_path = product_vector_path
        self.embeddings = self._init_embeddings()
        self.vector_store = self.load_vector_store()
        self.all_products = self._load_all_products()
        self.id_to_product = {p["id"]: p for p in self.all_products if "id" in p}

        # 初始化工具
        self.search_tool = self._create_search_tool()
        self.product_tool = self._create_product_tool()

    def _init_embeddings(self):
        """初始化嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def load_vector_store(self):
        """加载向量数据库"""
        if not os.path.exists(self.product_vector_path):
            print(f"警告：商品知识库路径不存在 {self.product_vector_path}")
            return None
        try:
            return FAISS.load_local(
                self.product_vector_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"加载商品知识库失败：{e}")
            return None

    def _parse_product_info(self, content):
        """解析商品信息字符串为字典"""
        product_info = {}
        for line in content.split("\n"):
            if "商品ID：" in line:
                product_info["id"] = line.split("：")[1].strip()
            if "名称：" in line:
                product_info["name"] = line.split("：")[1].strip()
            if "规格：" in line:
                product_info["spec"] = line.split("：")[1].strip()
            if "介绍：" in line:
                product_info["description"] = line.split("：")[1].strip()
            if "价格：" in line:
                product_info["price"] = line.split("：")[1].strip()
            if "活动：" in line:
                product_info["activity"] = line.split("：")[1].strip()
        return product_info

    def _load_all_products(self):
        """加载所有商品信息"""
        if not self.vector_store:
            return []
        all_docs = list(self.vector_store.docstore._dict.values())
        products = []
        for doc in all_docs:
            products.append(self._parse_product_info(doc.page_content))
        return products

    def _create_search_tool(self):
        """创建商品搜索工具"""

        def search_products_by_keyword(keyword: str):
            """通过关键词、名称或描述搜索商品"""
            if not self.vector_store:
                return "商品知识库未加载"

            # 使用向量数据库进行语义搜索
            results = self.vector_store.similarity_search(keyword, k=2)

            if not results:
                return f"未找到与 '{keyword}' 相关的商品"

            response = f"找到与 '{keyword}' 相关的商品：\n"
            for i, doc in enumerate(results, 1):
                product_info = self._parse_product_info(doc.page_content)
                response += f"{i}. 商品ID: {product_info.get('id', '未知')}\n"
                response += f"   名称: {product_info.get('name', '未知')}\n"
                response += f"   规格: {product_info.get('spec', '未知')}\n"
                response += f"   价格: {product_info.get('price', '未知')}\n"
                response += f"   活动: {product_info.get('activity', '无')}\n\n"

            return response.strip()

        return StructuredTool.from_function(
            func=search_products_by_keyword,
            name="search_products",
            description="通过关键词、名称、描述或类别搜索商品，参数为keyword（如'衬衫'、'红色连衣裙'）"
        )

    def _create_product_tool(self):
        """创建商品查询工具"""

        def query_product_info(product_id: str):
            """查询商品详情（含规格）"""
            if not self.vector_store:
                return "商品知识库未加载"

            if product_id in self.id_to_product:
                p = self.id_to_product[product_id]
                return (f"商品ID {product_id} 信息：\n"
                        f"- 名称：{p.get('name', '未知')}\n"
                        f"- 规格：{p.get('spec', '未知')}\n"
                        f"- 介绍：{p.get('description', '未知')}\n"
                        f"- 价格：{p.get('price', '未知')}\n"
                        f"- 活动：{p.get('activity', '未知')}")
            return f"未找到商品ID {product_id} 的信息"

        return StructuredTool.from_function(
            func=query_product_info,
            name="query_product",
            description="查询商品详情（含规格），参数为product_id（商品ID，如001）"
        )

    def get_tools(self):
        """提供给接入Agent的工具接口"""
        return [self.search_tool, self.product_tool]