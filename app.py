import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

load_dotenv()  # loads .env into os.environ (and won’t override existing env vars by default) [web:45]

# --- Required ENV (no defaults) ---
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

OLLAMA_URL   = os.environ["OLLAMA_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

# --- Neo4j connection ---
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
)  # Neo4jGraph params [web:20]

# --- Local Ollama LLM ---
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
)  # ChatOllama usage [web:2]

transformer = LLMGraphTransformer(llm=llm)  # usage [web:23]

# # Example run
text = """
Ada Lovelace was a mathematician who worked with Charles Babbage on the Analytical Engine.
She is often credited as the first computer programmer.
"""

# docs = [Document(page_content=text)]
# graph_docs = transformer.convert_to_graph_documents(docs)  # method exists [web:23]
# print(f"Nodes:{graph_docs[0].nodes}")
# print(f"Relationships:{graph_docs[0].relationships}")
# graph.add_graph_documents(graph_docs, include_source=True)  # method on Neo4jGraph [web:20]

#result=graph.query('MATCH (n) RETURN COUNT(n) as TOTAL')
result=graph.query('MATCH (n) DETACH DELETE n')
print(result)