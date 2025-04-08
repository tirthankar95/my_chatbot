from chain_base import Chains, MIN_CHAT_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
import os 
import dotenv
from typing import List, Dict
import logging 
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()

class Chain_General(Chains):
    """class Chain_General: is used to answer generic user queries."""
    def __init__(self, model_name: str = "qwen2.5-7b-instruct-q4_0.gguf", \
                 embed_model: str = "N/A", embed_model_dir: str = "N/A", \
                 chroma_db_dir: str = "N/A") -> None:
        super().__init__()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_api_base = "http://localhost:8000/v1"
        self.model_name = model_name
        self.local_dir = embed_model_dir
        self.repo_id = embed_model
        ## BUILD LangChain 
        self.init_model()
        # Get Prompt.
        self.prompt()
        # Build chain.
        self.chain_fn = self.prmpt | self.model | StrOutputParser() 
    
    def init_model(self):
        self.model = ChatOpenAI(
            api_key = self.openai_api_key,
            base_url = self.openai_api_base,
            model_name = self.model_name
        )

    def prompt(self):
        self.prmpt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert AI assistant, who answers user's questions."""),
                   MessagesPlaceholder(variable_name="history"),
                   ("user", "{query}")
            ]
        )
    
    def call_chain(self, query: str, history: List[Dict]) -> str:
        '''
        This function calls the lang chain for a specific child class.
        Args:
            query (str): The user query that will be passed to the LLM chain.
            history List[Dict]: The chat history that will be passed to the LLM chain.
        Return:
            output (str): The output string from the LLM chain.
        '''
        history_sz = min(MIN_CHAT_HISTORY, len(history))
        try:
            logging.info('--' * 50 + '\n')
            logging.info(f'history: {history}\n')
            logging.info(f'query: {query}\n')
            history_to_take = history[-history_sz:]
            history_to_take = [{'role': hist['role'], 'content': hist['content']} for hist in history_to_take]
            return self.chain_fn.invoke({"query": query, "history": history_to_take})
        except Exception as e:
            return str(e)

if __name__ == "__main__":
    import gradio as gr 
    general_chain = Chain_General()
    gr.ChatInterface(general_chain.call_chain, type = "messages").launch(share = False)
    