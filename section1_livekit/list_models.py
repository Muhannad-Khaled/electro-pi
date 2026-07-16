import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("Models available on your API key:\n")
for model in client.models.list():
    if "generateContent" in model.supported_actions:
        print(f"  {model.name}")