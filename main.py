import os,json
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_qwq import ChatQwen
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, TypedDict, Literal
from saver import save_vectorstore

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

llm = ChatQwen(
        model=os.getenv("DASH_MODEL_NAME"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )

def router_node(state: RouterState) -> Dict:
    """路由决策节点"""
    
    # 路由分类器
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个高效的rag系统问题分类器。请将问题分类为：save或retrieve.`save`表示对数据库进行编辑的操作，`retrieve`表示从数据库中检索信息的操作。"),
        ("human", "问题内容：{input}.文档内容：{documents}")
    ])
    # 使用 LLM 进行分类
    classifier_chain = classifier_prompt | llm | RunnableLambda(
        lambda x: x.content.lower().strip()
    )
    # 获取分类结果
    route = classifier_chain.invoke({"input": state["input"],"documents":state["documents"]})
    # 映射到有效路由类型
    valid_routes = ["save","retrieve"]
    if route not in valid_routes:
        route = "unknown"
    return {"route": route}

def save_node(state: RouterState) -> Dict:
    """保存文档节点"""
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个高效的文档切片处理器。请根据{state['input']}，决定文档的切片处理方式与切片大小，\
        {state['input']}中可能会对特定文档要求特定切片方式跟大小，也可能会规定所有文档的切片方式与大小，也可能不做规定，你应该{state['doc_info']}中获取每个文档的标题，并对每个文档的切片方式与大小单独进行判断。\
        切片处理方式：`fixed`表示固定大小切片，`recursive`表示递归切片，`semantic`表示语义切片，`document`表示按文档格式切片,\
        如果{state['input']}中没有提到切片处理方式，你应该设置为默认的`semantic`；\
        切片大小：你应该根据{state['input']}决定，如果{state['input']}没有提到，你应该设置为默认的{200}，如果{state['input']}中提到了，你应该设置为{state['input']}中提到的大小。\
        切片处理方式与切片大小的设置应该是独立的，你可以单独设置每个文档的切片处理方式与切片大小；\
        你的输出格式如下：\
        {'切片方式': ['fixed','recursive',...,'document'], '切片大小': [200,100,...,100]}"),
        ("human", "问题内容：{input}.文档内容：{documents}")
    ])
    chunk_chain = chunk_prompt | llm
    raw_chunks = chunk_chain.invoke({"input": state["input"],"documents":state["documents"]})
    raw = raw_chunks.content if hasattr(raw_chunks, 'content') else str(raw_chunks.content)
    try:
        json_data = json.loads(raw)
        chunks = json_data["切片方式"]
        size = json_data["切片大小"]
    except json.JSONDecodeError:
        return {"output": f"文件分块失败，chunk_llm响应内容无法解析为 JSON: {raw}"}

    return save_vectorstore(state["documents"],chunks,size)



def retrieve_node(state: RouterState) -> Dict:
    """检索文档节点"""
    try:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )
        
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
