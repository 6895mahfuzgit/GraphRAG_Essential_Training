import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph,GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()  # loads .env into os.environ (and won’t override existing env vars by default) [web:45]

# --- Required ENV (no defaults) ---
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

OLLAMA_URL   = os.environ["OLLAMA_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

# --- Local Ollama LLM ---
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
)  # ChatOllama usage [web:2]



transformer = LLMGraphTransformer(llm=llm)  # usage [web:23]

# --- Neo4j connection ---
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
)  # Neo4jGraph params [web:20]




CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to 
answer questions about skiing.
Convert the user's question based on the schema.
When you are presented with query properties such as id's like "grass skiing", 
be sure to convert the first letter to capital case, such as "Grass Skiing" 
before you run the Cypher query.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)


cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

# Updated query strings for the Ada Lovelace text
query_string = """
    Was the mathematician Ada Lovelace associated 
    with the development of the Analytical Engine?
"""

# Execute the chain with the specific query string
cypher_chain.invoke({"query": query_string})

# Execute the chain with a general inquiry
cypher_chain.invoke({"query": "Tell me about the computer programmer."})

# # # Example run
# text = """
# Ada Lovelace was a mathematician who worked with Charles Babbage on the Analytical Engine.
# She is often credited as the first computer programmer.
# """

# docs = [Document(page_content=text)]
# graph_docs = transformer.convert_to_graph_documents(docs)  # method exists [web:23]
# print(f"Nodes:{graph_docs[0].nodes}")
# print(f"Relationships:{graph_docs[0].relationships}")
# graph.add_graph_documents(graph_docs, include_source=True)  # method on Neo4jGraph [web:20]

# result=graph.query('MATCH (n) RETURN COUNT(n) as TOTAL')
# #result=graph.query('MATCH (n) DETACH DELETE n')
# print(result)