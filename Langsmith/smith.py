import dotenv
import os 
from langchain_openai import ChatOpenAI 

dotenv.load_dotenv()
openai_api_key = "NA"
openai_api_base = "http://localhost:8000/v1"
model_name = "qwen2.5-7b-instruct-q4_0.gguf"
model = ChatOpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base,
    model_name = model_name
)
model.invoke("Tell me a joke.")