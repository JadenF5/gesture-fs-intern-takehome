"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base
from typing import Callable
import argparse


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], list[dict]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    # TODO: implement this (~6-8 lines)
    if not question or not question.strip():
        return {"answer": "Ask a non-empty question.", "sources": []}
    docs = vector_store.similarity_search(question, k=3)
    if not docs:
        return {"answer": "I don't have enough information to answer that.", "sources": []}
    text = " ".join([chunk.page_content for chunk in docs])
    prompt = PROMPT_TEMPLATE.format(context=text, question=question)
    result = llm(prompt)
    answer = result[0]["generated_text"]

    return {"answer": answer, "sources": [chunk.page_content for chunk in docs]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """

    parser = argparse.ArgumentParser(description="Interactive Q&A Pipeline")
    parser.add_argument("--query", type=str, help="Question to ask the knowledge base and exit immediately")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # TODO: implement this (~10-12 lines)
    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' does not exist. Please create it and add .txt files.")
        return
    try:
        vector_store = build_knowledge_base(data_dir)
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        return
    llm = get_llm()

    if args.query:
        result = ask_question(vector_store, llm, args.query)
        print("\nSources:")
        for source in result["sources"]:
            print(source)
        print("Answer:")
        print(result["answer"])
        return

    while True:
        input_question = input("Ask a question (or type 'quit' to exit): ")
        if input_question.lower() == "quit":
            break
        if not input_question:
            continue
        result = ask_question(vector_store, llm, input_question)
        print("\nSources:")
        for source in result["sources"]:
            print(source)
        print("Answer:")
        print(result["answer"])

if __name__ == "__main__":
    main()