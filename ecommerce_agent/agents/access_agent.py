from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class AccessAgent:
    def __init__(self, llm_config, embedding_model_path, product_vector_path):
        # 初始化大模型
        self.llm = self._init_llm(llm_config)

        # 初始化子Agent
        from .order_agent import OrderAgent
        from .product_agent import ProductAgent

        self.order_agent = OrderAgent()
        self.product_agent = ProductAgent(
            embedding_model_path=embedding_model_path,
            product_vector_path=product_vector_path
        )

        # 初始化工具和执行器
        self.tools = self._init_tools()
        self.memory = self._init_memory()
        self.prompt = self._init_prompt()
        self.executor = self._init_executor()

    def _init_llm(self, config):
        """初始化大语言模型"""
        return ChatOpenAI(
            openai_api_key=config["api_key"],
            base_url=config["base_url"],
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )

    def _init_tools(self):
        """初始化工具集合"""
        return [
            self.order_agent.get_tool()
        ] + self.product_agent.get_tools()

    def _init_memory(self):
        """初始化对话记忆"""
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            k=10  # 保留最近10轮对话
        )

    def _init_prompt(self):
        """初始化提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            你是智能协调Agent，负责处理用户关于订单和商品的所有问题，流程如下：

            【核心能力】
            1. 理解用户问题：判断是订单问题、商品问题，还是复合问题（需同时查询订单和商品）。
            2. 工具调用规则：
               - 订单问题（含订单ID、物流、金额、签收时间等）→ 调用query_order工具，参数为order_id。
               - 商品问题：
                 * 已知商品ID，查询详情 → 调用query_product工具，参数为product_id。
                 * 未知商品ID，通过名称、类别、描述查询 → 调用search_products工具，参数为keyword。
               - 复合问题（如“订单12345中的T恤规格”）→ 先调用query_order获取商品ID，再调用query_product查询每个ID的详情，最后筛选出目标商品。
            3. 记忆利用：{chat_history} 包含历史对话，重复问题直接用记忆回答，无需重复调用工具。
            4. 结果整合：多工具调用后，需将结果汇总、筛选，用自然语言简洁回答用户。

            【工具调用格式】
            只能使用以下工具：query_order、query_product、search_products，严格用以下格式：
            {{"name": "工具名", "parameters": {{"参数名": "值"}}}}

            【注意】
            - 若用户问题缺少必要参数（如未提供订单ID），需询问用户补充。
            - 复合问题需分步处理，先获取必要信息，再逐步解决。
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def _init_executor(self):
        """初始化Agent执行器"""
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            memory=self.memory,
            early_stopping_method="force"
        )

    def handle_question(self, question: str) -> str:
        """处理用户问题的入口"""
        # 添加/no_think关闭推理过程（根据模型特性调整）
        question_with_no_think = f"{question} /no_think"
        result = self.executor.invoke({"input": question_with_no_think})
        return result["output"].replace("", "").strip()