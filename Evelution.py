from ragas.evaluation import evaluate 
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAI
from datasets import Dataset 
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from Base import LLM , collection , token_corpus , chunks 

def Hybrid_connection(query: str):

    result = collection.query(
        query_texts=[query],
        n_results=3
    )

    document = result['documents'][0]
    distances = result['distances'][0]
    print(distances ,)
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

    return top_docs 

def generation(question: str, context_list: list):

    if not context_list:
        return "NOT RELATED CONTENTS"

    context_str = "\n\n".join(context_list)

    prompt = f"""
You are a helpful AI assistant for a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY the information
provided in the retrieved context.

Rules:
1. Use only the given context to answer the question.
2. Do not use outside knowledge or make up information.
3. If the context does not contain enough information to answer the question,
   say: "I don't have enough information in the provided context."
4. Give a clear, accurate, and concise answer.
5. Do not mention the retrieval process unless necessary.
6. If multiple pieces of context are relevant, combine them into one coherent answer.

Retrieved Context:
{context_str}

User Question:
{question}

Final Answer:
"""

    answer = LLM.invoke(prompt)
    return answer.content

Question = "What is the name of India's lower house of Parliament?"
Ground_truth = "Lok Sabha"

retrive = Hybrid_connection(query=Question)
LLM_answer = generation(question=Question , context_list=retrive)

data = {
        "user_input":[Question] , 
        "retrieved_contexts":[retrive] , 
        "response":[LLM_answer] , 
        "reference":[Ground_truth]
    }
dataset = Dataset.from_dict(data)

from langchain_community.embeddings import HuggingFaceEmbeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

result = evaluate(
    dataset=dataset , 
    embeddings=embeddings , 
    metrics=[Faithfulness(),AnswerRelevancy(),ContextPrecision(),ContextRecall()] , 
    llm=LLM , 
    raise_exceptions=False
)
print(result)
