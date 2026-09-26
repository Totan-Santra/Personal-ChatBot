# import and get the pdf 
import warnings 
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader 
# load document  
pdf = PyPDFLoader('GK_Questions.pdf')

try:
    document = pdf.load()
except Exception as e :
    print(str(e)) 

# split Data 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_split = RecursiveCharacterTextSplitter(chunk_size=1600 , chunk_overlap = 180)
split_document = text_split.split_documents(document)

chunks = [i.page_content for i in split_document]
metadata = [i.metadata for i in split_document]

#  create unique IDs

import hashlib
ids = [hashlib.md5(chunk.encode('utf-8')).hexdigest() for chunk in chunks]

# Hybrid Create 
from rank_bm25 import BM25Okapi 
token_corpus = BM25Okapi([i.split() for i in chunks])
token_corpus

# Create VectorDB 
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction 
import chromadb 

# Client 
client = chromadb.PersistentClient(path="./VectorDB")

# Collection 
collection = client.get_or_create_collection(name="Collection",embedding_function=DefaultEmbeddingFunction())

try:
    if collection.count()!= len(chunks):
        collection.add(
        ids=ids ,
        documents=chunks , 
        metadatas=metadata
    )
except Exception as e :
    print(str(e))

# LLM Connection  
from langchain_groq import ChatGroq 
from dotenv import load_dotenv 
import os 
load_dotenv()
try:
    key = os.getenv("GROQ_API_KEY")
    
    # Groq model
    LLM = ChatGroq(model="openai/gpt-oss-120b",api_key=key)
except Exception as e :
    print(str(e))

from langchain_core.tools import tool

@tool
def Hybrid_connection(query: str):
    """
    Retrieve relevant information from the local knowledge base
    using hybrid search that combines BM25 keyword retrieval
    with semantic vector similarity search.

    Use this tool when the user's question is related to information
    contained in the stored documents or knowledge base.

    This tool is especially useful when the query contains:
    - specific keywords, names, terms, or phrases
    - exact facts or technical terminology
    - concepts that may require semantic similarity
    - questions where both keyword matching and semantic understanding
      are useful

    Do not use this tool for simple mathematical calculations,
    general conversation, or current/latest information from the web.
    """

    result = collection.query(
        query_texts=[query],
        n_results=3
    )

    document = result['documents'][0]
    distances = result['distances'][0]
    dense_chunks = []
    threshold = 1.9

    for i , doc in zip(distances , document):
        if i < threshold:
            dense_chunks.append(doc)

    score = token_corpus.get_scores(query.split())

    def get_near_keyword(score , k=10):
        index = list(enumerate(score))
        index_sorted = sorted(index , key = lambda x : x [1] , reverse=True)
        return [i for i , _ in index_sorted[:10]]

    keyword_search = get_near_keyword(score , k=10)

    get_chunks = [chunks[i] for i in keyword_search]

    rrf_token_corpus = {}

    for rank , docs in enumerate(dense_chunks):
        rrf_token_corpus[docs] = rrf_token_corpus.get(docs , 0)+1/(rank+60)
    for rank , docs in enumerate(get_chunks):
        rrf_token_corpus[docs] = rrf_token_corpus.get(docs , 0)+1/(rank+60)
    marge = sorted(rrf_token_corpus.items() , key=lambda x:x[1] , reverse=True)
    top_docs = [doc for doc , i in marge[:5]]

    if not top_docs:
        return "NOT RELATED CONTENTS"
    return "\n\n".join(top_docs)

# calculator 
import numexpr 
@tool 
def calculator(execution:str):
    """Perform mathematical calculations from a given expression.

    Use this tool when the user asks for arithmetic or numerical
    calculations such as addition, subtraction, multiplication,
    division, percentages, powers, or other basic mathematical
    expressions.

    Do not use this tool for general questions, document retrieval,
    or current/latest information from the web."""
    try :
        response = numexpr.evaluate(execution)
        return str(response)
    except Exception as e :
        return str(e)

# Web Search Fall Back 
from langchain_community.tools import DuckDuckGoSearchRun
@tool
def web_search(query: str) -> str:
    """
    Search the internet for current, recent, or publicly available information.

    Use this tool when the user asks for:
    - current or latest information
    - recent news or events
    - information that may have changed over time
    - current facts, prices, trends, or updates
    - information that is not available in the local knowledge base

    Do not use this tool for:
    - mathematical calculations; use the calculator tool
    - questions that can be answered from the local document knowledge base;
      use the hybrid_retrieve tool instead
    - general conversation or greetings

    Input:
        query: A clear search query describing the information to find.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e :
        return str(e)