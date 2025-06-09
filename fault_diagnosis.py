# -*- coding: utf-8 -*-
import os
import json
import dashscope
import numpy as np
from dashscope import Generation
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# ===================== 配置区 =====================
CONFIG = {
    "ali_api_key": os.getenv("ALI_API_KEY"),       # 从环境变量获取阿里云API Key
    "llm_model": "qwen-turbo",                     # 可选qwen-turbo/qwen-plus/qwen-max
    "chunk_size": 500,                            # 更小的分块大小
    "chunk_overlap": 50,
    "top_k": 3,                                   # 减少返回文档数量
}

# 设置阿里云SDK
dashscope.api_key = CONFIG["ali_api_key"]

# ===================== 自定义嵌入模块 =====================
class CustomEmbeddings(Embeddings):
    """符合Langchain接口的自定义嵌入实现"""
    def __init__(self):
        super().__init__()
        # 创建一个简单的虚拟嵌入模型
        self.embedding_size = 128  # 更小的嵌入维度
        self.vocab = {}
        self.next_id = 1
        self.embeddings_cache = {}

    def embed_text(self, text):
        """生成文本嵌入"""
        # 如果已经缓存过，直接返回
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # 为每个单词创建ID并生成随机嵌入
        words = text.split()
        word_ids = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            word_ids.append(self.vocab[word])

        # 生成一个简单的平均嵌入向量
        embeddings = np.zeros(self.embedding_size)
        for word_id in word_ids:
            np.random.seed(word_id)  # 保证一致性
            embeddings += np.random.randn(self.embedding_size)
        embeddings /= len(word_ids)

        # 归一化向量
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings /= norm

        # 缓存结果
        self.embeddings_cache[text] = embeddings
        return embeddings

    def embed_documents(self, texts):
        """实现Langchain所需的接口方法 - 批量嵌入文档"""
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text):
        """实现Langchain所需的接口方法 - 嵌入查询"""
        return self.embed_text(text)

# ===================== 1. 知识库构建模块 =====================
class KnowledgeBaseBuilder:
    """故障知识库构建器（支持增量更新）"""
    def __init__(self, config):
        self.config = config
        # 使用自定义嵌入模型
        self.embedder = CustomEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

    def process_data(self, historical_data):
        """数据处理：分块和向量化"""
        documents = []
        metadatas = []

        for record in historical_data:
            # 构建文档描述 = 现象 + 分类 + 措施 
            text = f"故障现象: {record['symptom']}\n分类: {record['category']}\n措施: {record['solution']}"
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({
                    "source": record.get("source", "unknown"),
                    "timestamp": record.get("timestamp", datetime.now().isoformat()),
                    "category": record["category"]
                })

        # 创建向量数据库 
        print(f"创建向量数据库，包含 {len(documents)} 个文档片段...")
        vector_db = Chroma.from_texts(
            documents, 
            metadatas=metadatas,
            embedding=self.embedder
        )
        return vector_db, documents

# ===================== 2. 简化检索模块 =====================
class SimplifiedRetriever:
    """简化的混合检索器"""
    def __init__(self, vector_db, documents, config):
        self.config = config
        self.vector_db = vector_db
        self.documents = documents

        # 创建BM25检索器
        print("初始化BM25检索器...")
        self.bm25_retriever = BM25Retriever.from_texts(documents)
        self.bm25_retriever.k = config["top_k"]

    def retrieve(self, query):
        """执行混合检索（简化版本）"""
        # 向量检索
        try:
            vector_results = self.vector_db.similarity_search(query, k=self.config["top_k"])
            vector_contents = [doc.page_content for doc in vector_results]
        except Exception as e:
            print(f"向量检索错误: {e}")
            vector_contents = []

        # BM25检索
        try:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            bm25_contents = [doc.page_content for doc in bm25_results]
        except Exception as e:
            print(f"BM25检索错误: {e}")
            bm25_contents = []

        # 融合结果并去重
        all_contents = list(set(vector_contents + bm25_contents))
        print(f"检索到 {len(all_contents)} 个相关文档片段")

        # 简单返回top_k
        return all_contents[:self.config["top_k"]]

# ===================== 3. RAG推理模块（适配阿里API） =====================
class AliyunLLMChain:
    """封装阿里云大模型调用"""
    def __init__(self, config):
        self.config = config
        self.model = config["llm_model"]

    def generate(self, prompt):
        """调用通义千问API"""
        try:
            print("调用阿里大模型API...")
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                result_format='message'
            )
            return response["output"]["choices"][0]["message"]["content"]
        except Exception as e:
            return f"大模型调用失败: {str(e)}"

class FaultDiagnosisSystem:
    """故障定位核心系统"""
    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config
        self.llm = AliyunLLMChain(config)

        # 更结构化的提示模板
        self.prompt_template = """
        你是一个资深运维专家，请根据告警信息和历史故障数据诊断问题：

        <当前告警>
        描述: {alert_desc}
        指标: {alert_metrics}
        日志: {alert_logs}

        <相关历史案例>
        {context}

        请按以下结构输出分析结果：
        1. 故障类别: 从历史案例中选择最匹配的分类
        2. 诊断依据: 结合当前告警和历史案例相似点分析
        3. 紧急措施: 建议的第一步操作
        4. 备注: 任何额外注意事项
        """

    def build_prompt(self, alert_desc, alert_metrics, alert_logs, context):
        """构建完整提示"""
        # 格式化历史案例
        context_str = "\n".join([f"📝 {doc}" for doc in context])
 
        print("\n======= 提示内容 =======")
        print(self.prompt_template.format(
            alert_desc=alert_desc,
            alert_metrics=alert_metrics,
            alert_logs=alert_logs,
            context=context_str
        ))
        print("=" * 40)

        return self.prompt_template.format(
            alert_desc=alert_desc,
            alert_metrics=alert_metrics,
            alert_logs=alert_logs,
            context=context_str
        )

    def diagnose(self, alert_desc, alert_metrics, alert_logs):
        """执行故障诊断"""
        # 检索相关文档
        print(f"检索告警: {alert_desc}...")
        context_docs = self.retriever(alert_desc)

        # 构建提示
        print("构建提示...")
        prompt = self.build_prompt(
            alert_desc, 
            alert_metrics, 
            alert_logs,
            context_docs
        )
   
        # 调用阿里大模型
        return self.llm.generate(prompt)

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    print("正在初始化故障诊断系统...")

    # 0. 确保阿里API密钥存在
    if not CONFIG["ali_api_key"]:
        print("错误：未设置ALI_API_KEY环境变量！")
        print("请执行：export ALI_API_KEY=您的阿里API密钥")
        exit(1)

    # 1. 构建知识库（示例数据）
    historical_data = [
        {
            "symptom": "数据库响应延迟突增至5s，CPU使用率90%",
            "category": "数据库死锁",
            "solution": "1. 检查锁等待链\n2. 终止阻塞事务\n3. 优化索引"
        },
        {
            "symptom": "内存占用每小时增长2%，Full GC频繁",
            "category": "内存泄漏",
            "solution": "1. 生成堆转储\n2. 分析内存对象\n3. 修复引用链"
        },
        {
            "symptom": "服务接口成功率从99.9%骤降至60%，错误码ConfigNotFount",
            "category": "配置错误",
            "solution": "1. 检查最近部署的配置\n2. 回滚最新配置\n3. 验证配置加载流程"
        },
        {
            "symptom": "接口流量超过阈值95%，TCP重传率增加300%",
            "category": "网络阻塞",
            "solution": "1. 扩容带宽\n2. 优化流量调度\n3. 分析异常流量源"
        }
    ]

    # 1.1 初始化知识库
    print("正在构建知识库...")
    kb_builder = KnowledgeBaseBuilder(CONFIG)
    vector_db, documents = kb_builder.process_data(historical_data)
    print(f"知识库构建完成，包含 {len(documents)} 个文档片段")

    # 2. 初始化检索器
    print("初始化混合检索器...")
    retriever = SimplifiedRetriever(vector_db, documents, CONFIG)

    # 2.1 创建检索函数
    def retrieve_function(query):
        return retriever.retrieve(query)

    # 3. 初始化诊断系统
    print("初始化诊断引擎...")
    diagnosis_system = FaultDiagnosisSystem(
        retrieve_function, 
        CONFIG
    )
    print("系统初始化完成，准备接受诊断请求")
    print("=" * 60)

    # 4. 测试用例（模拟故障告警）
    test_cases = [
        {
            "desc": "订单服务响应超时率超过60%",
            "metrics": "DB连接池活跃连接100%，SQL执行平均耗时2.5s",
            "logs": "发现大量'Lock wait timeout'错误日志"
        },
        {
            "desc": "用户服务API失败率突增",
            "metrics": "最近部署后失败率从0.1%升至40%",
            "logs": "出现大量'ConfigNotFount'错误"
        },
        {
            "desc": "服务内存使用持续增长",
            "metrics": "JVM内存占用每小时增长3%，GC频率增加",
            "logs": "观察到频繁的Full GC日志"
        }
    ]

    # 5. 执行诊断
    for i, test_alert in enumerate(test_cases):
        print(f"\n===== 测试用例 {i+1} =====")
        print(f"告警描述: {test_alert['desc']}")
        print(f"相关指标: {test_alert['metrics']}")
        print(f"日志信息: {test_alert['logs']}")

        print("\n正在分析中...")
        # 执行诊断
        result = diagnosis_system.diagnose(
            test_alert["desc"],
            test_alert["metrics"],
            test_alert["logs"]
        )

        print("\n===== 诊断结果 =====")
        print(result)
        print("=" * 100)

    print("所有测试用例执行完毕")
