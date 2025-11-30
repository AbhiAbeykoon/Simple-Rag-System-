import os
import google.generativeai as genai

# PASTE YOUR KEY HERE FOR TESTING
os.environ["GOOGLE_API_KEY"] = "Enter Your API Key 

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Checking available models for your API key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:

    print(f"Error connecting: {e}")
