from chain_base import Chains, MIN_CHAT_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from save_chat import Save_Chat
import logging 
from models import LM_Models
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
from time import time 
class Chain_General(Chains):
    """Chain_General class is used to answer generic user queries that can't be answered using other chains."""
    def __init__(self, session_id: str) -> None:
        super().__init__()
        ## Get Chat Object
        self.chat_obj = Save_Chat(collection_name = session_id)
        ## BUILD LangChain 
        self.prompt()
        self.model = LM_Models().lm_model
        self.chain_fn = self.prmpt | self.model | StrOutputParser() 

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
        resp = ""
        history_to_take = history[-min(MIN_CHAT_HISTORY, len(history)):]
        logging.info('-g' * 30 + '\n')
        logging.info(f'history: {history_to_take}\n')
        logging.info(f'query: {query}\n')
        resp = self.chain_fn.invoke({"query": query, "history": history_to_take})
        history.append([
            {
                "role": "user", 
                "content": self.prmpt.invoke({"query": query, "history": history_to_take}).to_string()
            },
            {
                "role": "assistant", 
                "content": resp
            }
        ])
        self.chat_obj.insert_many(history[-2:])
        return resp

if __name__ == "__main__":
    import gradio as gr 
    general_chain = Chain_General()
    gr.ChatInterface(general_chain.call_chain, type = "messages").launch(share = False)
    