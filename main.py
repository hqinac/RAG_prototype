import os,json,ast
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_qwq import ChatQwen
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, TypedDict, Literal
from saver import save_vectorstore, check_db_exist

load_dotenv()
# 定义状态对象
class RouterState(TypedDict):
    # 输入文本与文件
    input: str
    documents: list
    doc_info: list
    # 路由决策
    route: Literal["save","retrieve","unknown"]
    # 处理结果
    output: str

def get_llm():
    """获取LLM实例"""
    return ChatQwen(
        model=os.getenv("DASH_MODEL_NAME"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )

def router_node(state: RouterState) -> Dict:
    """路由决策节点"""
    llm = get_llm()
    
    # 路由分类器
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个高效的RAG系统问题分类器。请将用户问题分类为：save 或 retrieve

分类标准：
- save：涉及向数据库/知识库添加、保存、存储、上传、导入文档的操作
  示例：将文档加入数据库、保存文件到知识库、上传文档、导入数据、存储信息
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
    route = classifier_chain.invoke({"input": state["input"]})
    # 映射到有效路由类型
    valid_routes = ["save","retrieve"]
    if route not in valid_routes:
        route = "unknown"
    return {"route": route}

def save_node(state: RouterState) -> Dict:
    """保存文档节点"""
    llm = get_llm()
    
    # 构建system prompt，避免f-string中的花括号问题
    system_prompt = (
        f"你是一个高效的文档切片处理器。请根据问题内容，决定文档的切片处理方式与切片大小，"
        f"问题内容中可能会对特定文档要求特定切片方式跟大小，也可能会规定所有文档的切片方式与大小，也可能不做规定，"
        f"你应该从文档内容中按顺序获取每个文档的标题，并对每个文档的切片方式与大小单独进行判断。"
        "切片处理方式：`fixed`表示固定大小切片，`recursive`表示递归切片，`semantic`表示语义切片，`document`表示按文档格式切片，"
        f"如果问题内容中没有提到切片处理方式，你应该设置为默认的`semantic`；"
        f"切片大小：你应该根据问题内容决定，如果问题内容没有提到，你应该设置为默认的200，"
        f"如果问题内容中提到了，你应该设置为问题内容中提到的大小。"
        "切片处理方式与切片大小的设置应该是独立的，你可以单独设置每个文档的切片处理方式与切片大小；"
        "你的输出格式如下：\n"
        "{{'切片方式': ['fixed','recursive',...,'document'], '切片大小': [200,100,...,100]}}"
        "输出中切片方式与切片大小的顺序应该与文档内容中各文档的顺序一一对应。"
    )
    
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "问题内容：{input}.文档内容：{doc_info}")
    ])
    chunk_chain = chunk_prompt | llm
    raw_chunks = chunk_chain.invoke({"input": state["input"],"doc_info":state["doc_info"]})
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
    except KeyError as e:
        return {"output": f"文件分块失败，响应内容缺少必要字段 {e}: {raw}"}
    temp_output = save_vectorstore(state["documents"],chunks,size)

    return {"output": temp_output}



def retrieve_node(state: RouterState) -> Dict:
    """检索文档节点"""
    llm = get_llm()
    system_prompt = (
        f"你是一个高效的知识库检索器，请够根据{state['input']}提出的问题决定检索策略。"
        "共有三种检索方式：`HyDE`、`FaissBert`、`BM25Rerank`；"
        "`HyDE`表示生成假设文档进行检索，`FaissBERT`表示Faiss近似最近邻搜索，`BM25Rerank`表示语义重排；"
        f"如果{state['input']}中提到了对检索方式的要求，你应该按照要求选择策略，如果没有，你应该根据问题的性质决定。"
        "如果问题描述模糊但需要深度理解，如”解释量子隧穿效应“，你应该选择`HyDE`检索方式；"
        "如果问题描述具体，包含明确专业术语，如”请告诉我Python的GIL机制“，你应该选择`BM25Rerank`检索方式；"
        "其余一般情况应该选择`FaissBERT`检索方式。"
        "请只回答HyDE或FaissBERT或BM25Rerank,不要包含其他内容。"
    )
    strategy_chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "用户问题：{input}")
    ]) | llm | RunnableLambda(
        lambda x: x.content.lower().strip()
    )

    strategy = strategy_chain.invoke({"input": state["input"]})
    if strategy not in ["hyde","faissbert","bm25rerank"]:
        strategy = "faissbert"
    URI = os.getenv("URI", "./vector_db")
    if not check_db_exist(URI):
        return {"output": "知识库为空，请先上传文档到知识库。"}
    
    try:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        retrieve(strategy,state["input"],embeddings)
        # 这里应该加载已保存的向量数据库
        # 暂时返回一个示例回答
        query = state["input"]
        answer = f"根据您的问题 '{query}'，我正在知识库中搜索相关信息..."
        
        return {"output": answer}
        
    except Exception as e:
        return {"output": f"检索过程中出现错误: {str(e)}"}



def unknown_node(state: RouterState) -> Dict:
    """未知路由节点"""
    return {"output": "我无法理解您提出的问题，请重新描述您的需求。我是一个知识库问答系统，您可以上传文档到知识库并自定义切片方式与大小，也可以提出问题让我在知识库中搜索回答。"}


def create_graph():
    workflow = StateGraph(RouterState)
    workflow.add_node("router", router_node)
    workflow.add_node("save", save_node)
    workflow.add_node("retrieve", retrieve_node)
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
    workflow.add_edge("retrieve", END)
    workflow.add_edge("unknown", END)
    
    return workflow.compile()

graph = create_graph()
