````markdown
# 🤖 Personal Chatbot

An AI-powered **Personal Chatbot** built with **LangChain, LangGraph, Hybrid RAG, Web Search, Calculator Tool, Groq LLM, and Streamlit**.

The chatbot can understand user questions and intelligently decide whether to use the local knowledge base, web search, or calculator.

---

## 🚀 Features

- 🤖 Personal AI Chatbot
- 🧠 LangGraph-based agent workflow
- 📚 Hybrid RAG
- 🔎 Semantic + Keyword Search
- 🌐 Web Search
- 🧮 Calculator Tool
- 💬 Conversational Chat Interface
- 🧵 Thread-based conversation memory
- 💾 SQLite-based LangGraph checkpointing
- 📊 RAG evaluation using Ragas
- 🖥️ Streamlit Web Interface
- 🔐 Environment variable based API keys
- 🦜 LangChain integration
- ⚡ Groq LLM

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     User Query      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LangGraph      │
                    │       Agent         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ Hybrid RAG │   │ Web Search │   │ Calculator │
       └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
             │                │                │
             ▼                ▼                ▼
       Local Documents     Internet       Mathematical
       BM25 + Vector       Search          Calculation
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Final Answer    │
                    └─────────────────────┘
````

---

# 🧠 Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Programming language            |
| Streamlit             | Web application                 |
| LangChain             | LLM application framework       |
| LangGraph             | Agent workflow and tool calling |
| Groq                  | LLM provider                    |
| ChromaDB              | Vector database                 |
| BM25                  | Keyword-based retrieval         |
| Sentence Transformers | Text embeddings                 |
| Tavily                | Web search                      |
| SQLite                | Conversation checkpointing      |
| Ragas                 | RAG evaluation                  |
| GitHub                | Version control                 |

---

# 📂 Project Structure

```text
Personal-Chatbot/
│
├── app.py
├── Base.py
├── create_database.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── Chromadb/
│
├── conversation_memory.sqlite
│
├── GK_Questions.pdf
└── NIPS-2017-attention-is-all-you-need-Paper.pdf
```

> `.env`, `Chromadb/`, SQLite files, and private documents should not be committed to GitHub.

---

# 🔧 Main Components

## 1. Hybrid RAG

The chatbot uses a hybrid retrieval approach combining:

* Dense vector search
* BM25 keyword search
* Reciprocal Rank Fusion (RRF)

This allows the system to handle both:

* Semantic questions
* Exact keyword-based questions

### Retrieval Flow

```text
User Query
     │
     ├───────────────┐
     │               │
     ▼               ▼
Vector Search      BM25 Search
     │               │
     ▼               ▼
Semantic Rank     Keyword Rank
     │               │
     └───────┬───────┘
             ▼
      RRF Combination
             │
             ▼
       Top Documents
```

---

# 🌐 Web Search

The chatbot uses web search when information may be:

* Current
* Recent
* Time-sensitive
* Not available in the local knowledge base

The web search tool is powered by **Tavily**.

---

# 🧮 Calculator

The calculator tool handles mathematical expressions such as:

```text
125 * 48
100 / 4
(25 + 15) * 2
```

The LangGraph agent can automatically select the calculator tool when required.

---

# 🧠 LangGraph Agent

LangGraph controls the chatbot workflow.

The basic workflow is:

```text
START
  │
  ▼
LLM
  │
  ├── Tool required ──► ToolNode
  │                       │
  │                       ▼
  │                      LLM
  │
  └── No tool required ─► END
```

Available tools:

```text
Hybrid RAG
Web Search
Calculator
```

---

# 💬 Conversation Memory

The project uses LangGraph checkpointing with SQLite.

Example:

```python
config = {
    "configurable": {
        "thread_id": "id_1"
    }
}
```

The `thread_id` identifies a conversation thread.

The same thread ID can be used to continue a conversation.

Example:

```python
response = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Hello"
            )
        ]
    },
    config=config
)
```

---

# 📊 RAG Evaluation

The project uses **Ragas** to evaluate the RAG pipeline.

Evaluation metrics include:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

Example evaluation result:

```text
Faithfulness      : 1.0000
Answer Relevancy  : 0.9648
Context Precision : 1.0000
Context Recall    : 1.0000
```

These results are based on the evaluation question used during development.

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

If LangSmith tracing is enabled:

```env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Agentic-Research-Assistant
```

Never upload `.env` to GitHub.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Totan-Santra/Agentic-Research-Assistant.git
```

Move into the project:

```bash
cd Agentic-Research-Assistant
```

---

## 2. Create Virtual Environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖥️ User Interface

The application provides:

```text
🤖 Personal Chatbot

Ask me anything...
```

The sidebar contains:

```text
⚙️ Assistant

● System Online

Available Tools

📚 Hybrid RAG
🌐 Web Search
🧮 Calculator

💡 Example Questions

🗑️ Clear Conversation
```

---

# 🧪 Example Questions

### General Question

```text
What is the capital of India?
```

### Knowledge Base Question

```text
Explain Transformer architecture.
```

### Web Search

```text
What is the latest information about AI?
```

### Calculator

```text
What is 125 × 48?
```

### RAG

```text
What is RAG?
```

---

# 🔄 Agent Decision Making

The chatbot decides which tool should handle the query.

### Example 1

```text
User:
What is 125 × 48?

        ↓

Calculator Tool

        ↓

Answer
```

### Example 2

```text
User:
Explain the Transformer architecture.

        ↓

Hybrid RAG

        ↓

Knowledge Base

        ↓

Answer
```

### Example 3

```text
User:
What is the latest AI news?

        ↓

Web Search

        ↓

Internet

        ↓

Answer
```

---

# 📈 Future Improvements

Possible future improvements include:

* 👤 User-specific thread IDs
* 🔐 Authentication
* 🧠 Long-term memory
* 📄 Upload custom documents
* 🔎 Better hybrid retrieval
* 📊 Advanced RAG evaluation
* 🧑‍💻 Multi-agent architecture
* 📝 Conversation history
* 🎤 Voice input
* 🔊 Voice output
* 🌍 Multilingual support
* ☁️ Cloud deployment
* 📱 Mobile-friendly interface
* 📈 LangSmith monitoring
* 🔄 Streaming responses

---

# ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

Basic deployment flow:

```text
GitHub Repository
        │
        ▼
Streamlit Community Cloud
        │
        ▼
Add API Secrets
        │
        ▼
Deploy
        │
        ▼
Public Web Application
```

Required secrets:

```text
GROQ_API_KEY
TAVILY_API_KEY
LANGSMITH_API_KEY
```

---

# 🔐 Security

Do not expose API keys in source code.

Use:

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
```

and add `.env` to `.gitignore`.

---

# 📚 Learning Concepts Covered

This project demonstrates practical implementation of:

```text
Python
   ↓
LangChain
   ↓
LLM
   ↓
RAG
   ↓
Hybrid RAG
   ↓
Tool Calling
   ↓
LangGraph
   ↓
Agentic Workflow
   ↓
Memory
   ↓
Ragas Evaluation
   ↓
Streamlit Deployment
```

---

# 👨‍💻 Author

**Totan Santra**

GitHub:

https://github.com/Totan-Santra

Project:

https://github.com/Totan-Santra/Agentic-Research-Assistant

---

# ⭐ Project Summary

**Personal Chatbot** is an AI-powered assistant that combines:

```text
LLM
+
Hybrid RAG
+
Web Search
+
Calculator
+
LangGraph
+
Memory
+
Streamlit
```

to create an intelligent chatbot capable of selecting the appropriate tool for different types of user queries.

---

## ❤️ If you find this project useful

Feel free to ⭐ the repository and explore the project.

```

You can save this directly as **`README.md`** in your project root and push it to GitHub.
```
