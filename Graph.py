# import modules
from langgraph.graph import StateGraph, START  
from typing import TypedDict , Annotated 
from langgraph.graph.message import add_messages 
from Base import Hybrid_connection , web_search , calculator , LLM
from langgraph.checkpoint.memory import MemorySaver 
# memory 
memory = MemorySaver()

# State build 
class state(TypedDict):
    messages:Annotated[list,add_messages]

build = StateGraph(state)

#tools

tools = [Hybrid_connection , web_search , calculator]

#tool bind
LLM_bind = LLM.bind_tools(tools)

def llm_tool_bind(state:state):

    response = LLM_bind.invoke(state["messages"])

    return{
        "messages": response
    }

# config add 
config = {'configurable':{'thread_id':'id_1'}}
# Node build 
from langgraph.prebuilt import ToolNode , tools_condition 
build.add_node('llm_tools' , llm_tool_bind)
build.add_node('tools',ToolNode(tools))

# Edge 
build.add_edge(START , 'llm_tools')
build.add_conditional_edges(

    'llm_tools',

    tools_condition ,

    {"tools": "tools", "__end__": "__end__"},

)
build.add_edge('tools','llm_tools')

graph = build.compile(checkpointer=memory)

# testing 
from langchain_core.messages import HumanMessage 
response = graph.invoke({
    'messages':
    HumanMessage('what city called city of joy ?')
},
config=config
)

result = response['messages'][-1].content
print(result)