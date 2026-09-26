
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Personal Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Main container */
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .app-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 1rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Tool cards */
    .tool-card {
        padding: 12px;
        border-radius: 10px;
        background-color: #1f2937;
        margin-bottom: 8px;
        border: 1px solid #374151;
    }

    .tool-name {
        font-weight: 600;
        font-size: 0.95rem;
    }

    .tool-description {
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 3px;
    }

    /* Status */
    .status-online {
        color: #22c55e;
        font-weight: 600;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .app-title {
            font-size: 1.7rem;
        }

        .app-subtitle {
            font-size: 0.85rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_graph():

    from Base import (
        Hybrid_connection,
        web_search,
        calculator,
        LLM,
    )
    from langgraph.checkpoint.memory import MemorySaver 
    # memory 
    memory = MemorySaver()
    from langgraph.graph import StateGraph, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing import TypedDict, Annotated

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    build = StateGraph(State)

    tools = [
        Hybrid_connection,
        web_search,
        calculator,
    ]

    LLM_bind = LLM.bind_tools(tools)

    def llm_tool_bind(state: State):

        response = LLM_bind.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    build.add_node(
        "llm_tools",
        llm_tool_bind
    )

    build.add_node(
        "tools",
        ToolNode(tools)
    )

    build.add_edge(
        START,
        "llm_tools"
    )

    build.add_conditional_edges(
        "llm_tools",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "__end__",
        }
    )

    build.add_edge(
        "tools",
        "llm_tools"
    )

    graph = build.compile(checkpointer=memory)

    return graph


# Load graph

graph = load_graph()


if "messages" not in st.session_state:

    st.session_state.messages = []


# Front Header

st.markdown(
    "# 🤖 Personal Chatbot"
)


with st.sidebar:

    st.title("⚙️ Assistant")

    st.markdown(
        '<div class="status-online">● System Online</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Available Tools")

    st.markdown(
        """
        <div class="tool-card">
            <div class="tool-name">📚 Hybrid RAG</div>
            <div class="tool-description">
                Searches the local knowledge base using
                semantic + keyword retrieval.
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-name">🌐 Web Search</div>
            <div class="tool-description">
                Searches the internet for current information.
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-name">🧮 Calculator</div>
            <div class="tool-description">
                Performs mathematical calculations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("💡 Example Questions")

    examples = [
        "What is the capital of India?",
        "What city is called the City of Joy?",
        "What is 125 × 48?",
        "Explain Transformer architecture.",
        "What is RAG?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True
        ):

            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        if "pending_question" in st.session_state:
            del st.session_state.pending_question

        st.rerun()


for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)


pending_question = st.session_state.pop(
    "pending_question",
    None
)

user_input = st.chat_input(
    "Ask me anything..."
)


# Example button input
config = {'configurable':{'thread_id':'id_1'}}


if pending_question:

    user_input = pending_question


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                # LangGraph invocation

                response = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    } ,
                    config=config
                )

                final_message = response["messages"][-1]

                answer = final_message.content

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                error_message = (
                    "⚠️ Something went wrong.\n\n"
                    f"```text\n{str(e)}\n```"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        padding:30px 0 10px 0;
        font-size:0.8rem;
    ">

    Powered by LangGraph • LangChain • RAG • Groq

    </div>
    """,
    unsafe_allow_html=True
)

