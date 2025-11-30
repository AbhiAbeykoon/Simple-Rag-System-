import argparse
import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
#from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

GOOGLE_API_KEY = "Enter Your API Key"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    print("Loading database...")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("Searching for relevant documents...")
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("No matching documents found in Chroma DB.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Sending prompt")
    print(prompt)

    print("Sending prompt to Google Gemini...")
    
    # We use "flash" because it is optimized for speed
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY
    )

    # --- START TIMING ---
    start_time = time.time()
    
    response = model.invoke(prompt)
    response_text = response.content # <--- IMPORTANT: Extract text from object
    
    end_time = time.time()
    # --- END TIMING ---
    # -----------------------

    elapsed_seconds = end_time - start_time
    print(f"Response received in {elapsed_seconds:.2f} seconds.")
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":

    main()
