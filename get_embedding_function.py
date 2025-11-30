from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_API_KEY = "AIzaSyAPgpFXiPfrClTfoSWsVYRc62M66SzBhZ0"

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    return embeddings