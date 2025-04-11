from chain_base import Chains, MIN_CHAT_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import logging 
from models import LM_Models
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
class Chain_General(Chains):
    """Chain_General class is used to answer generic user queries that can't be answered using other chains."""
    def __init__(self) -> None:
        super().__init__()
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
        try:
            logging.info('--' * 50 + '\n')
            logging.info(f'history: {history}\n')
            logging.info(f'query: {query}\n')
            return self.chain_fn.invoke({"query": query, "history": history})
        except Exception as e:
            return str(e)

if __name__ == "__main__":
    import gradio as gr 
    general_chain = Chain_General()
    gr.ChatInterface(general_chain.call_chain, type = "messages").launch(share = False)
    