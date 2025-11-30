from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = "Enter Your API key"

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )

    return embeddings
