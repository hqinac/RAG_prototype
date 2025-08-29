import os,json,ast
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_qwq import ChatQwen
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, TypedDict, Literal, Any
from saver import save_vectorstore, check_db_exist
from retriever import retrieve
from evaluator import RetrievalEvaluator, GenerationEvaluator
from cache_manager import get_llm, get_embeddings
from detailed.utils import outputtest_file

load_dotenv()

class SafeStrOutputParser(BaseOutputParser[str]):
    """安全的字符串输出解析器，确保始终返回字符串"""
    
    def parse(self, text: Any) -> str:
        """解析输出为字符串"""
        if hasattr(text, 'content'):
            return str(text.content)
        elif isinstance(text, str):
            return text
        else:
            return str(text)
    
    @property
    def _type(self) -> str:
        return "safe_str"
# 定义状态对象
class RouterState(TypedDict):
    # 输入文本与文件
    input: str
    documents: list
    temp_doc_names: list
    doc_list: list
    # 路由决策
    route: Literal["save","retrieve","unknown"]
    knowledgebase: list[Document]
    query: str
    answer: str
    # 处理结果
    output: str

async def router_node(state: RouterState) -> Dict:
    """路由节点，决定是保存还是检索"""
    llm = get_llm()
    
    # 路由分类器
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能路由器，负责分析用户输入并决定操作类型。

请根据用户输入的内容，判断用户想要执行的操作类型：

- save：涉及上传、保存、存储、添加文档的操作  
  示例：上传文件、保存文档、添加到知识库、存储信息

- retrieve：涉及从数据库/知识库查询、检索、搜索、获取信息的操作  
  示例：查找相关信息、搜索文档内容、回答问题、获取数据

请只回答 save 或 retrieve，不要包含其他内容。"""),
        ("human", "用户问题：{input}")
    ])
    # 使用 LLM 进行分类
    classifier_chain = classifier_prompt | llm | RunnableLambda(
        lambda x: x.content.lower().strip()
    )
    # 获取分类结果
    route = await classifier_chain.ainvoke({"input": state["input"]})
    # 映射到有效路由类型
    valid_routes = ["save","retrieve"]
    if route not in valid_routes:
        route = "unknown"
    return {"route": route}

async def save_node(state: RouterState) -> Dict:
    """保存文档节点"""
    llm = get_llm()
    
    # 构建system prompt，避免f-string中的花括号问题
    system_prompt = (
        "你是一个高效的文档切片处理器。**默认情况下，所有文档都使用`default`自定义切片方式。** 请根据问题内容，为每个文档单独决定切片处理方式与切片大小,同时根据文档预览内容，决定文档的语言。"
        "重要规则："
        "1. 仔细分析问题内容，识别是否对特定文档有特殊要求"
        "2. 文档内容与文档预览内容的顺序是一一对应的。"
        "2. 如果问题中明确指定某个文档使用特定切片方式或切片大小（如'某某.md按文件格式切片，切片大小设置为100'），则只对该文档应用指定方式与指定切片大小"
        "3. **重要：如果问题中没有明确指定切片方式，必须使用默认的`default`方式。**"
        "4. 必须为文档内容中的每个文档都输出对应的切片方式和大小,并且决定每个文档的语言。"
        "5. 根据语义仔细分辨问题的要求，例如提到‘文档‘不一定是要按照文档格式切片，注意区分。"
        "\n 文档语言只包括`cn`,`en`两种，其中cn代表中文，en代表英语。"
        "\n切片处理方式说明："
        "- `semantic`: 语义切片"
        "- `fixed`: 固定大小切片"
        "- `recursive`: 递归切片" 
        "- `document`: 按文档格式切片（仅当明确要求按markdown标题结构切片时使用）"
        "- `default`: 自定义结构切片"
        "\n处理逻辑："
        "- **首要原则：除非问题中明确指定其他切片方式，否则所有文档都使用`default`方式**"
        "- 如果问题中提到特定文档名并指定切片方式、大小，只对该文档使用指定方式大小，其他文档使用默认值"
        "- 如果问题中对所有文档统一要求，则所有文档使用相同方式大小"
        "- 如果同时对所有文档进行要求并对特定文档指定方式大小，则对对应的特定文档使用指定方式大小，其他文档使用统一指定的方式大小"
        "- 文档内容中可能包含多个文档，每个文档的切片方式和大小必须独立指定"
        "- **默认行为：如果问题中没有明确要求，所有文档必须使用`default`方式**"
        "- 切片大小默认为200。"
        "\n输出格式："
        "{{'切片方式': ['default','default','default'], '切片大小': [200,200,200], '使用语言': ['cn','cn','en']}}"
        "输出中切片方式、切片大小与使用语言的顺序必须与文档内容中各文档的顺序一一对应。"
    )
    preview = []
    for document in state["documents"]:
        preview.append(document.page_content[:100])
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "问题内容：{input},文档内容：{temp_doc_names},文档预览：{preview}")
    ])
    chunk_chain = chunk_prompt | llm
    raw_chunks = await chunk_chain.ainvoke({"input": state["input"],"temp_doc_names":state["temp_doc_names"],"preview":preview})
    raw = raw_chunks.content if hasattr(raw_chunks, 'content') else str(raw_chunks.content)
    try:
        # 首先尝试标准JSON解析
        json_data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # 如果JSON解析失败，尝试使用ast.literal_eval解析Python字典格式
            json_data = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {"output": f"文件分块失败，chunk_llm响应内容无法解析: {raw}"}
    
    try:
        chunks = json_data["切片方式"]
        size = json_data["切片大小"]
        language = json_data["使用语言"]
        print(chunks)
        print(size)
    except KeyError as e:
        return {"output": f"文件分块失败，响应内容缺少必要字段 {e}: {raw}"}
    temp_output,embedded = await save_vectorstore(state["documents"],chunks,size,state["doc_list"],language)
    state["doc_list"].extend(embedded)
    return {"output": temp_output}



async def retrieve_node(state: RouterState) -> Dict:
    """检索文档节点"""
    llm = get_llm()
    system_prompt = (
        "你是一个高效的知识库检索器，请够根据用户问题提出的问题决定检索策略。"
        "共有四种检索方式：`HyDE`、`FaissBert`、`BM25Rerank`、`default`；"
        "`HyDE`表示生成假设文档进行检索，`FaissBERT`表示Faiss近似最近邻搜索，`BM25Rerank`表示语义重排,'default'表示将以上三种策略结合使用；"
        "如果用户问题中提到了对检索方式的要求，你应该按照要求选择策略，如果没有，你应该选择`default`。"
        "请只回答HyDE或FaissBERT或BM25Rerank或default，,不要包含其他内容。"
    )
    strategy_chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "用户问题：{input}")
    ]) | llm | RunnableLambda(
        lambda x: x.content.lower().strip()
    )

    strategy = await strategy_chain.ainvoke({"input": state["input"]})
    if strategy not in ["hyde","faissbert","bm25rerank","default"]:
        strategy = "default"
    URI = os.getenv("URI", "./saved_files")
    if not check_db_exist():
        return {"output": "知识库为空，请先上传文档到知识库。"}
    
    guery_prompt = (
        "你是一个高效的知识库检索器，请根据输入内容提取需要检索的问题与可能的过滤条件。"
        
        "**核心原则：严格区分过滤条件和问题内容**"
        "- 过滤条件：仅来自用户明确指定的文件范围要求"
        "- 问题内容：用户想要查询的具体问题"
        "- **绝对禁止**：将问题内容中的主题词汇用于文件过滤"
        
        "**过滤条件判断规则（严格执行）：**"
        "1. **默认无过滤**：如果输入中没有明确的文件范围指示词，文件过滤必须设置为空列表[]"
        "2. **文件范围指示词识别**：只有包含以下明确指示词时才进行过滤："
        "   - '在...文件中' / 'in...file'"
        "   - '从...文档中' / 'from...document'"
        "   - '...相关文件' / '...related files'"
        "   - '查看...类型的文档' / 'check...type documents'"
        "3. **问题内容排除**：问题中的疾病名、技术名等主题词汇绝不用于文件过滤"
        
        "**关键示例（重点学习）：**"
        "正确示例："
        "输入内容：'What is GIL in python?' → 文件过滤:[] （无文件范围指示词）"
        "输入内容：'如何治疗阑尾？' → 文件过滤:[] （无文件范围指示词）"
        "输入内容：'Python的GIL机制是什么？' → 文件过滤:[] （无文件范围指示词）"
        "输入内容：'please tell me what is GIL in python, retrieve from python.md?'→ 文件过滤:['python.md'] （有明确指示词）"
        "输入内容：'请在肿瘤科相关文件中查询阑尾治疗' → 文件过滤:['肿瘤科疾病治疗方案汇总.md'] （有明确指示词）"

        
        "错误示例："
        "输入内容：'Tell me treatments of throat cancer.' → 文件过滤:['肿瘤科疾病治疗方案汇总.md'] （错误：将问题内容用于过滤）"

        
        "**语义匹配表（仅在有明确文件范围指示词时使用）：**"
        "**重要提醒：此表仅用于文件范围指示词的匹配，绝不用于问题内容的匹配**"
        "- 当用户说'在外科相关文件中'时：外科、手术、Surgery、Surgical → ['外科疾病治疗方案汇总.md']"
        "- 当用户说'在肿瘤科相关文件中'时：肿瘤、癌症、Oncology、Cancer → ['肿瘤科疾病治疗方案汇总.md']"
        "- 当用户说'在儿科相关文件中'时：儿科、小儿、Pediatric → ['儿科疾病治疗方案汇总.md']"
        "**严禁示例：**"
        "- '小孩发烧怎么办' → 文件过滤:[] （'小孩'是问题内容，不是文件范围指示词）"
        "- '癌症如何治疗' → 文件过滤:[] （'癌症'是问题内容，不是文件范围指示词）"
        "- '外科手术注意事项' → 文件过滤:[] （'外科手术'是问题内容，不是文件范围指示词）"
        
        "**标准示例：**"
        "输入内容：'请用HyDE方式对python.md文件进行检索，告诉我Python的GIL机制'"

        ""
        "文件列表：['python.md', 'python2.md', 'GIL.md']"
        "分析：有明确文件指定'python.md文件'"
        "输出：{{\"文件过滤\": [\"Python.md\"], \"问题\": \"Python的GIL机制\"}}"
        
        "输入：'请在python相关的文件中为我搜索Python的GIL机制'"
        "文件列表：['python.md', 'python2.md', 'GIL.md']"
        "分析：有文件范围指示词'python相关的文件中'"
        "输出：{{\"文件过滤\": [\"python.md\",\"python2.md\"], \"问题\": \"Python的GIL机制\"}}"
        
        "输入：'为我搜索Python的GIL机制'"
        "分析：无任何文件范围指示词"
        "输出：{{\"文件过滤\": [], \"问题\": \"Python的GIL机制\"}}"
        
        "输入：'What is GIL in python?'"
        "分析：无任何文件范围指示词，'GIL in python'是问题内容不是过滤条件"
        "输出：{{\"文件过滤\": [], \"问题\": \"What is GIL in python?\"}}"
        
        "输入：'小孩发烧怎么办'"
        "分析：无任何文件范围指示词，'小孩发烧'是问题内容不是过滤条件"
        "输出：{{\"文件过滤\": [], \"问题\": \"小孩发烧怎么办\"}}"
        
        "输入：'请在儿科相关文件中查询小孩发烧的处理方法'"
        "分析：有明确文件范围指示词'在儿科相关文件中'"
        "输出：{{\"文件过滤\": [\"儿科疾病治疗方案汇总.md\"], \"问题\": \"小孩发烧的处理方法\"}}"
        
        "你的输出格式必须严格遵循json格式："
        "{{\"文件过滤\": [], \"问题\": \"提取的问题内容\"}}"
    )
    Query_prompt = ChatPromptTemplate.from_messages([
        ("system", guery_prompt),
        ("human", "输入内容：{input}, 文件列表：{doc_list}")
    ])

    query_chain = Query_prompt | llm
    raw_querys = await query_chain.ainvoke({"input": state["input"],"doc_list":state["doc_list"]})
    rawq = raw_querys.content if hasattr(raw_querys, 'content') else str(raw_querys.content)
    try:
        # 首先尝试标准JSON解析
        json_data = json.loads(rawq)
    except json.JSONDecodeError:
        try:
            # 如果JSON解析失败，尝试使用ast.literal_eval解析Python字典格式
            json_data = ast.literal_eval(rawq)
        except (ValueError, SyntaxError):
            return {"output": f"文件分块失败，query_llm响应内容无法解析: {rawq}"}
    
    try:
        filters = json_data["文件过滤"]
        query = json_data["问题"]
    except KeyError as e:
        return {"output": f"文件分块失败，响应内容缺少必要字段 {e}: {rawq}"}
    

    #return {"output": temp_output}
    try:
        print(f"开始检索过程，策略: {strategy}")
        print(f"查询内容: {query}")
        print(f"文件过滤: {filters}")
        
        print("Embeddings初始化成功")
        
        # 检索文档
        try:
            print("开始调用retrieve函数...")
            docs, Strategy = await retrieve(strategy, query, filters)
            print(f"检索完成，策略: {Strategy}")
            if not docs:
                return {"output": "未检索到相关文档，请尝试调整搜索关键词或检查知识库内容。"}
        except ValueError as ve:
            print(f"检索ValueError: {str(ve)}")
            return {"output": f"检索失败: {str(ve)}"}
        except Exception as re:
            print(f"检索Exception: {str(re)}")
            return {"output": f"检索器初始化失败: {str(re)}"}
        
        contents = [doc.page_content for doc in docs]
        names = [doc.metadata.get("source", "未知来源") for doc in docs]
        print(f"检索到的文档来源: {names}")
        print(f"检索到的文档数量: {len(docs)}")
        
        # 生成回答
        try:
            print("开始生成回答...")
            answer_prompt = (
                "你是一个高效的回答编辑器，能根据检索到的相关文件内容为问题生成对应的回答或对已有答案进行修改。"
                "如果存在回答，根据你读到的文档内容对回答进行修改，否则根据文档内容生成回答。"
                "请严格按照文件内容与已有的回答生成与修改回答，不要添加任何额外的内容。"
                "你的回答使用的语言应该根据问题使用的语言。如果问题是中文，用中文回答，如果问题是英语，用英语回答。"
                "注意：只用返回你生成或修改后的回答内容，不用写“根据文档内容生成回答”或“根据文档内容修改回答”"

            )
            answer_chain = ChatPromptTemplate.from_messages([
                ("system", answer_prompt),
                ("human", "问题：{query}\n相关文档内容：{contents}\n已有回答：{answer}")
            ]) | llm | SafeStrOutputParser()
            
            # 调用链并获取结果
            answer = ""
            for content in contents:
                answer = await answer_chain.ainvoke({"query": query, "contents": content, "answer": answer})
            print(f"回答类型: {type(answer)}")
            print(f"最终回答: {answer}")
            
            if not answer or answer.strip() == "":
                return {"output": "回答生成失败：LLM返回空响应，请检查模型服务状态。"}
                
        except Exception as ae:
            print(f"回答生成异常: {str(ae)}")
            return {"output": f"回答生成失败: {str(ae)}"}
        outputtest_file(docs,"answer.md")
        return {
            "knowledgebase": docs,
            "output": f"已经以检索方式{Strategy}为问题{query}检索到相关文档，回答如下：\n{answer}\n, 答案来源为{names}",
            "answer": answer,
            "query": query
        }
        
    except Exception as e:
        print(f"检索过程中出现未知错误: {str(e)}")
        return {"output": f"检索过程中出现未知错误: {str(e)}"}


async def evaluate_node(state: RouterState) -> Dict:
    """评估节点"""
    if state["knowledgebase"] == []:
        return {"output": "检索失败，请检查原因"}
    if state["answer"] == "":
        return {"output": "回答生成失败，请检查原因"}
    retrieverevaluator = RetrievalEvaluator(get_embeddings())
    reeval = retrieverevaluator.evaluate(state["query"],state["knowledgebase"])
    reanswer = f"检索评估结果由三个指标组成：平均相似度、最大相似度、多样性分数，三个分数均由向量相似度评估，在0-1之间，越大说明向量越相似，多样性越高。分别为：\n平均相似度：{reeval['avg_similarity']}\n最大相似度：{reeval['max_similarity']}\n多样性分数：{reeval['diversity_score']}"
    generateevaluator = GenerationEvaluator(get_llm())
    geeval = await generateevaluator.evaluate(state["query"],state["answer"],state["knowledgebase"])
    geanswer = f"生成评估结果由三个指标组成：事实一致性、回答相关性、信息完整性，三个分数均在0-5之间，越大说明越符合要求。分别为：\n事实一致性：{geeval['faithfulness']}\n回答相关性：{geeval['relevance']}\n信息完整性：{geeval['completeness']}"
    answeroutput = state["output"]
    return {"output": answeroutput + "\n对回答的评估结果如下：\n" + reanswer + "\n" + geanswer}



def unknown_node(state: RouterState) -> Dict:
    """未知路由节点"""
    return {"output": "我无法理解您提出的问题，请重新描述您的需求。我是一个知识库问答系统，您可以上传文档到知识库并自定义切片方式与大小，也可以提出问题让我在知识库中搜索回答。"}


def create_graph():
    workflow = StateGraph(RouterState)
    workflow.add_node("router", router_node)
    workflow.add_node("save", save_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("unknown", unknown_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda x: x["route"],
        {
            "save": "save",
            "retrieve": "retrieve",
            "unknown": "unknown"
        }
    )
    workflow.add_edge("save", END)
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_edge("evaluate", END)
    workflow.add_edge("unknown", END)
    
    return workflow.compile()

graph = create_graph()
