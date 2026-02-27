import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.evaluation.qa import QAEvalChain  # classic API moved here in LangChain v1+ [web:114]

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

OLLAMA_URL = os.environ["OLLAMA_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

CYPHER_GENERATION_TEMPLATE = """
Write ONE Cypher query for this medical Neo4j graph.

Rules:
- Use only labels/relationships/properties from the schema.
- Every returned variable must be defined in MATCH/OPTIONAL MATCH.
- Prefer OPTIONAL MATCH.
- For text filtering, extract the relevant keyword from the question and use it as a literal string (do NOT use Cypher parameters like $question):
  WHERE toLower(d.text) CONTAINS toLower("extracted_keyword")
- Return at most 50 rows.
- Output ONLY Cypher.

Schema: {schema}
Question: {question}
Cypher:
"""



cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

def main():
    # Neo4jGraph has close() and is also a context manager (__enter__/__exit__) [web:130][web:132]
    with Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS, enhanced_schema=True) as g:
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)

        cypher_chain = GraphCypherQAChain.from_llm(
            llm,
            graph=g,
            cypher_prompt=cypher_generation_prompt,
            verbose=True,
            allow_dangerous_requests=True,
        )

        examples = [
            {
                "query": "What was discussed with the patient and family before the examination?",
                "answer": "The procedure of the examination, including possible complications, was clearly discussed with the patient and family.",
            },
            {
                "query": "What consent action was taken before the procedure?",
                "answer": "A consent form was signed by the patient and/or the family.",
            },
        ]

        predictions = []
        for ex in examples:
            resp = cypher_chain.invoke({"query": ex["query"]})
            # GraphCypherQAChain returns a dict; the final answer is under output key "result" by default. [web:131]
            predictions.append({"result": (resp.get("result") or "").strip()})

        # QAEvalChain expects examples with keys query/answer and predictions with key result by default. [web:126][web:93]
        eval_chain = QAEvalChain.from_llm(llm)
        graded = eval_chain.evaluate(examples, predictions)

        for i, g in enumerate(graded, start=1):
            print(f"\n--- Example {i} ---")
            print(g)

        # Best-effort cleanup for Ollama socket warnings if supported by your version.
        if hasattr(llm, "close"):
            llm.close()

if __name__ == "__main__":
    main()
