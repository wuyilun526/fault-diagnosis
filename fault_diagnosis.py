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

# 导入配置文件
from config import SYSTEM_CONFIG, HISTORICAL_DATA, TEST_CASES, validate_config

# ===================== 配置区 =====================
CONFIG = SYSTEM_CONFIG.copy()
CONFIG["ali_api_key"] = os.getenv("ALI_API_KEY")  # 从环境变量获取阿里云API Key

# 设置阿里云SDK
dashscope.api_key = CONFIG["ali_api_key"]

# ===================== 自定义嵌入模块 =====================
class CustomEmbeddings(Embeddings):
    """符合Langchain接口的自定义嵌入实现"""
    def __init__(self):
        super().__init__()
        # 使用更大的嵌入维度以提高表达能力
        self.embedding_size = 384  # 增加维度
        self.vocab = {}
        self.next_id = 1
        self.embeddings_cache = {}

    def embed_text(self, text):
        """生成文本嵌入 - 改进版本"""
        # 如果已经缓存过，直接返回
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # 文本预处理：标准化和分词
        processed_text = self._preprocess_text(text)
        
        # 为每个单词创建ID并生成随机嵌入
        words = processed_text.split()
        word_ids = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            word_ids.append(self.vocab[word])

        # 生成更复杂的嵌入向量
        embeddings = np.zeros(self.embedding_size)
        
        # 1. 词级别嵌入
        for word_id in word_ids:
            np.random.seed(word_id)  # 保证一致性
            word_embedding = np.random.randn(self.embedding_size)
            embeddings += word_embedding
        
        # 2. 考虑词序信息
        for i, word_id in enumerate(word_ids):
            np.random.seed(word_id + i * 1000)  # 位置编码
            position_embedding = np.random.randn(self.embedding_size) * 0.1
            embeddings += position_embedding
        
        # 3. 考虑文本长度
        length_factor = min(len(words) / 10.0, 1.0)  # 归一化长度
        embeddings *= (1 + length_factor * 0.2)
        
        # 归一化向量
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings /= norm

        # 缓存结果
        self.embeddings_cache[text] = embeddings
        return embeddings

    def _preprocess_text(self, text):
        """文本预处理"""
        # 1. 标准化
        text = text.lower()
        
        # 2. 保留重要标点符号
        text = text.replace('，', ',').replace('。', '.').replace('：', ':')
        
        # 3. 移除多余空格
        text = ' '.join(text.split())
        
        return text

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
            # 构建更结构化的文档描述
            structured_text = self._build_structured_document(record)
            
            # 分块处理
            chunks = self.text_splitter.split_text(structured_text)
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                
                # 提取关键词并转换为字符串
                symptom_keywords = self._extract_keywords(record["symptom"])
                solution_keywords = self._extract_keywords(record["solution"])
                semantic_tags = self._generate_semantic_tags(record)
                
                metadatas.append({
                    "source": record.get("source", "unknown"),
                    "timestamp": record.get("timestamp", datetime.now().isoformat()),
                    "category": record["category"],
                    "chunk_id": str(i),  # 转换为字符串
                    "total_chunks": str(len(chunks)),  # 转换为字符串
                    "symptom_keywords": ", ".join(symptom_keywords) if symptom_keywords else "",  # 转换为字符串
                    "solution_keywords": ", ".join(solution_keywords) if solution_keywords else "",  # 转换为字符串
                    "semantic_tags": ", ".join(semantic_tags) if semantic_tags else ""  # 转换为字符串
                })

        # 创建向量数据库 
        print(f"创建向量数据库，包含 {len(documents)} 个文档片段...")
        vector_db = Chroma.from_texts(
            documents, 
            metadatas=metadatas,
            embedding=self.embedder
        )
        return vector_db, documents

    def _build_structured_document(self, record):
        """构建结构化的文档内容"""
        # 使用更清晰的语义结构
        structured_parts = [
            f"故障现象: {record['symptom']}",
            f"故障分类: {record['category']}", 
            f"解决方案: {record['solution']}"
        ]
        
        # 添加语义标签
        semantic_tags = self._generate_semantic_tags(record)
        if semantic_tags:
            structured_parts.append(f"语义标签: {', '.join(semantic_tags)}")
        
        return "\n".join(structured_parts)

    def _extract_keywords(self, text):
        """提取关键词"""
        # 简单的关键词提取
        keywords = []
        important_words = ['接口', '成功率', '下降', '跌落', '依赖', '下游', 'DB', '慢查询', '超时', '错误']
        
        for word in important_words:
            if word in text:
                keywords.append(word)
        
        return keywords

    def _generate_semantic_tags(self, record):
        """生成语义标签"""
        tags = []
        
        # 基于症状生成标签
        symptom = record['symptom'].lower()
        if '成功率' in symptom and '下降' in symptom:
            tags.append('成功率异常')
        if '依赖' in symptom or '下游' in symptom:
            tags.append('依赖链问题')
        if 'db' in symptom or '数据库' in symptom:
            tags.append('数据库问题')
        if '超时' in symptom:
            tags.append('超时问题')
        if '内存' in symptom:
            tags.append('内存问题')
        
        return tags

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
        """执行混合检索（通用语义版本）"""
        # 1. 向量检索 - 使用更大的候选集
        try:
            vector_results = self.vector_db.similarity_search_with_score(
                query, 
                k=self.config["top_k"] * 3  # 获取更多候选
            )
            vector_candidates = [(score, doc.page_content, doc.metadata) for doc, score in vector_results]
        except Exception as e:
            print(f"向量检索错误: {e}")
            vector_candidates = []

        # 2. BM25检索
        try:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            bm25_candidates = [(0.5, doc.page_content, {}) for doc in bm25_results]  # 默认分数
        except Exception as e:
            print(f"BM25检索错误: {e}")
            bm25_candidates = []

        print(f"向量检索候选: {len(vector_candidates)} 个")
        print(f"BM25检索候选: {len(bm25_candidates)} 个")

        # 3. 融合候选集
        all_candidates = self._merge_candidates(vector_candidates, bm25_candidates)
        
        # 4. 重新排序和去重
        final_results = self._rerank_and_deduplicate(query, all_candidates)
        
        print(f"最终检索到 {len(final_results)} 个相关文档片段")
        
        return final_results

    def _merge_candidates(self, vector_candidates, bm25_candidates):
        """融合向量检索和BM25检索的候选结果"""
        merged = {}
        
        # 添加向量检索结果
        for score, content, metadata in vector_candidates:
            if content not in merged:
                merged[content] = {
                    'score': score,
                    'metadata': metadata,
                    'sources': ['vector']
                }
        
        # 添加BM25检索结果，如果已存在则提升分数
        for score, content, metadata in bm25_candidates:
            if content in merged:
                # 如果两个检索器都找到了，提升分数
                merged[content]['score'] = merged[content]['score'] * 1.2
                merged[content]['sources'].append('bm25')
            else:
                merged[content] = {
                    'score': score,
                    'metadata': metadata,
                    'sources': ['bm25']
                }
        
        return merged

    def _rerank_and_deduplicate(self, query, candidates):
        """重新排序和去重"""
        # 转换为列表并排序
        candidate_list = []
        for content, info in candidates.items():
            # 计算最终分数
            final_score = self._calculate_final_score(query, content, info)
            candidate_list.append((final_score, content))
        
        # 按分数降序排序
        candidate_list.sort(key=lambda x: x[0], reverse=True)
        
        # 去重并返回top_k
        seen_contents = set()
        final_results = []
        
        for score, content in candidate_list:
            if content not in seen_contents and len(final_results) < self.config["top_k"]:
                final_results.append(content)
                seen_contents.add(content)
        
        return final_results

    def _calculate_final_score(self, query, content, info):
        """计算最终相似度分数"""
        base_score = info['score']
        
        # 1. 基础分数
        final_score = base_score
        
        # 2. 多源检索奖励（如果多个检索器都找到了）
        if len(info['sources']) > 1:
            final_score *= 1.1
        
        # 3. 元数据匹配奖励
        if 'metadata' in info and info['metadata']:
            metadata = info['metadata']
            
            # 检查关键词匹配
            query_lower = query.lower()
            if 'symptom_keywords' in metadata and metadata['symptom_keywords']:
                # 将字符串关键词分割回列表
                keywords = [kw.strip() for kw in metadata['symptom_keywords'].split(',') if kw.strip()]
                keyword_matches = sum(1 for keyword in keywords 
                                    if keyword.lower() in query_lower)
                final_score += keyword_matches * 0.1
            
            # 检查语义标签匹配（如果有的话）
            if 'semantic_tags' in metadata and metadata['semantic_tags']:
                # 将字符串标签分割回列表
                tags = [tag.strip() for tag in metadata['semantic_tags'].split(',') if tag.strip()]
                tag_matches = sum(1 for tag in tags 
                                if tag.lower() in query_lower)
                final_score += tag_matches * 0.2
        
        return final_score

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
        1. 故障类别: 从历史案例中选择最匹配的分类，如果和历史案例的分类匹配度不高，则另外给出简洁的故障类别
        2. 故障对象: 如果定位到故障根因对象，请输出故障根因对象
        3. 诊断依据: 结合当前告警和历史案例相似点分析
        4. 紧急措施: 建议的第一步操作
        5. 备注: 任何额外注意事项
        """

    def build_prompt(self, alert_desc, alert_metrics, alert_logs, context):
        """构建完整提示"""
        # 格式化历史案例
        context_str = "\n".join([f"📝 {doc}" for doc in context])
 
        # print("\n======= 提示内容 =======")
        # print(self.prompt_template.format(
        #     alert_desc=alert_desc,
        #     alert_metrics=alert_metrics,
        #     alert_logs=alert_logs,
        #     context=context_str
        # ))
        # print("=" * 40)

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

    # 0. 验证配置
    try:
        validate_config()
    except ValueError as e:
        print(f"配置验证失败: {e}")
        exit(1)

    # 0.1 确保阿里API密钥存在
    if not CONFIG["ali_api_key"]:
        print("错误：未设置ALI_API_KEY环境变量！")
        print("请执行：export ALI_API_KEY=您的阿里API密钥")
        exit(1)

    # 1. 构建知识库（从配置文件加载数据）
    print("正在构建知识库...")
    kb_builder = KnowledgeBaseBuilder(CONFIG)
    vector_db, documents = kb_builder.process_data(HISTORICAL_DATA)
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

    # 4. 执行测试用例（从配置文件加载）
    for i, test_alert in enumerate(TEST_CASES):
        print("=" * 160)
        print(f"\n===== 测试用例 {i+1} =====")
        print(f"告警描述: {test_alert['desc']}")
        print(f"相关指标: {test_alert['metrics']}")
        print(f"日志信息: {test_alert['logs']}")
        if 'expected_category' in test_alert:
            print(f"预期分类: {test_alert['expected_category']}")

        print("\n正在分析中...")
        # 执行诊断
        result = diagnosis_system.diagnose(
            test_alert["desc"],
            test_alert["metrics"],
            test_alert["logs"]
        )

        print("\n===== 诊断结果 =====")
        print(result)
        print("=" * 160)

    print("所有测试用例执行完毕")
