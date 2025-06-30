'''import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,
    model_name="gpt-4o",  # or "gpt-35-turbo"
    temperature=0.7,
)

# Tools
search = DuckDuckGoSearchRun()
yahoo = YahooFinanceNewsTool()
tools = [search, yahoo]
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant analyzing a company’s stock. Use historical price data, financial performance, and current news sentiment to evaluate its potential. Conclude with a 2-line investment insight."
)

# Reasoning function
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build LangGraph
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
builder.set_finish_point("reasoner")
react_graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="Stock Analysis Assistant", layout="centered")
st.title("📈 Stock News & Analysis Assistant")

user_query = st.text_input("Enter a stock/company name or query", "")

if st.button("Analyze") and user_query.strip():
    with st.spinner("Analyzing..."):
        messages = [HumanMessage(content=user_query)]
        result = react_graph.invoke({"messages": messages})
        st.subheader("📊 Assistant Response")
        for m in result["messages"]:
            st.markdown(m.content)
'''

"""
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure LLM setup
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,
    model_name="gpt-4o",
    temperature=0.7,
)

# Tools
search = DuckDuckGoSearchRun()
yahoo = YahooFinanceNewsTool()
tools = [search, yahoo]
llm_with_tools = llm.bind_tools(tools)

# System instruction
sys_msg = SystemMessage(
    content="You are a helpful assistant analyzing a company’s stock. Use historical price data, financial performance, and current news sentiment to evaluate its potential. Conclude with a 2-line investment insight."
)

# Reasoning function node
def reasoner(state: MessagesState):
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    # Store reasoning step
    if "steps" not in state:
        state["steps"] = []
    state["steps"].append(("LLM", response.content))
    return {"messages": [response], "steps": state["steps"]}

# LangGraph flow setup
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
builder.set_finish_point("reasoner")
react_graph = builder.compile()

# ---- Streamlit App ----

st.set_page_config(page_title="💬 Stock Chat Assistant", layout="centered")
st.title("💬 Live Stock Chat Assistant with LangGraph")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reasoning_steps" not in st.session_state:
    st.session_state.reasoning_steps = []

# Display chat
for msg in st.session_state.chat_history:
    role, content = msg
    if role == "user":
        st.chat_message("user").markdown(content)
    else:
        st.chat_message("assistant").markdown(content)

# Chat input
if user_input := st.chat_input("Ask about a stock or company..."):
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # LangGraph call
    with st.spinner("Analyzing..."):
        messages = [HumanMessage(content=user_input)]
        result = react_graph.invoke({"messages": messages})
        response = result["messages"][-1].content
        steps = result.get("steps", [])

        # Show response
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))

        # Save steps for graph visualization
        st.session_state.reasoning_steps = steps

# Reasoning flow (graph visualization)
if st.session_state.reasoning_steps:
    st.subheader("🧠 LangGraph Reasoning Steps")
    for i, (step_type, detail) in enumerate(st.session_state.reasoning_steps):
        st.markdown(f"**Step {i+1}: {step_type}**")
        st.info(detail)
"""


import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.memory import ConversationBufferMemory

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure OpenAI setup
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,
    model_name="gpt-4o",
    temperature=0.7,
)

# Tools setup
search = DuckDuckGoSearchRun()
yahoo = YahooFinanceNewsTool()
tools = [search, yahoo]
llm_with_tools = llm.bind_tools(tools)

# System message for guidance
sys_msg = SystemMessage(content="You are a helpful assistant analyzing a company’s stock. Use historical price data, financial performance, and current news sentiment to evaluate its potential. Conclude with a 2-line investment insight.")

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Reasoner node
def reasoner(state: MessagesState):
    chat_history = memory.load_memory_variables({}).get("history", [])
    all_messages = [sys_msg] + chat_history + state["messages"]
    response = llm_with_tools.invoke(all_messages)

    memory.save_context({"input": state["messages"][-1].content}, {"output": response.content})

    if "steps" not in state:
        state["steps"] = []
    state["steps"].append(("LLM", response.content))
    return {"messages": [response], "steps": state["steps"]}

# Custom logging wrapper for ToolNode
class LoggingToolNode:
    def __init__(self, tools):
        self.tool_node = ToolNode(tools)

    def __call__(self, state):
        result = self.tool_node.invoke(state)
        tool_call = result["messages"][-1].content
        if "steps" not in state:
            state["steps"] = []
        state["steps"].append(("Tool", tool_call))
        return {**result, "steps": state["steps"]}

# LangGraph setup
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", LoggingToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
builder.set_finish_point("reasoner")
react_graph = builder.compile()

# Streamlit UI
#st.set_page_config(page_title="📊 Stock Chat Assistant", layout="centered")
st.title("💬 Live Stock Assistant with LangGraph")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reasoning_steps" not in st.session_state:
    st.session_state.reasoning_steps = []

# Function to show stock price chart
def show_stock_chart(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="1mo", interval="1d")
        fig = go.Figure(data=go.Scatter(x=data.index, y=data['Close'], mode='lines', name=ticker_symbol))
        fig.update_layout(title=f"{ticker_symbol} Stock Price (1M)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Could not load stock chart: {e}")

# Display chat history
for role, content in st.session_state.chat_history:
    st.chat_message("user" if role == "user" else "assistant").markdown(content)

# Handle new input
if user_input := st.chat_input("Ask about a stock or company..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Analyzing..."):
        messages = [HumanMessage(content=user_input)]
        result = react_graph.invoke({"messages": messages})
        response = result["messages"][-1].content
        steps = result.get("steps", [])

        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))
        st.session_state.reasoning_steps = steps

        # Auto-plot stock chart if symbol detected
        if "adani" in user_input.lower():
            show_stock_chart("ADANIPORTS.NS")
        elif "tcs" in user_input.lower():
            show_stock_chart("TCS.NS")
        elif "reliance" in user_input.lower():
            show_stock_chart("RELIANCE.NS")

# Show reasoning steps
if st.session_state.reasoning_steps:
    st.subheader("🧠 LangGraph Reasoning Trace")
    for i, (step_type, content) in enumerate(st.session_state.reasoning_steps):
        st.markdown(f"**Step {i+1}: {step_type}**")
        st.info(content)

# Export chat history
st.subheader("📥 Download Chat History")
chat_data = [{"role": r, "content": c} for r, c in st.session_state.chat_history]
df = pd.DataFrame(chat_data)
st.download_button("⬇️ Download as CSV", df.to_csv(index=False), file_name="chat_history.csv")
st.download_button("⬇️ Download as JSON", json.dumps(chat_data, indent=2), file_name="chat_history.json")

