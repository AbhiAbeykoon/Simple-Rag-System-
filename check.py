from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral")
response = model.invoke("Hello, how are you?")

if response:
    print(response)
else:
    print("No response")
