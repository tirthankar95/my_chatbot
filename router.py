from chain_base import MIN_CHAT_HISTORY
from chain_mongo import Chain_Mongo
from chain_general import Chain_General
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os 
import dotenv
import logging 
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()
from typing import List, Dict

class ChainRouter():
    def __init__(self, model_name: str = "qwen2.5-7b-instruct-q4_0.gguf", \
                 embed_model: str = "thenlper/gte-base", embed_model_dir: str = "./embed_model/", \
                 chroma_db_dir: str = "LLM_MONGO_1") -> None:
        super().__init__()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_api_base = "http://localhost:8000/v1"
        self.model_name = model_name
        self.init_model()
        self.prompt()
        self.model_name = model_name
        self.mongo_chain = Chain_Mongo(model_name=self.model_name, embed_model=embed_model, \
                                      embed_model_dir=embed_model_dir, chroma_db_dir=chroma_db_dir)
        self.general_chain = Chain_General(model_name=self.model_name)
        self.chain_fn = self.prmpt | self.model | StrOutputParser()
    
    def prompt(self):
        '''
        Prompt for the LLM.
        '''
        self.prmpt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""
You are a planner responsible for determining which chains to invoke in order to solve a user's query.
You are not allowed output anything else other than the chain names and the sequence in which they should be invoked.
These are the chains you can use:
1. {Chain_Mongo.__doc__}
2. {Chain_General.__doc__}
"""),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{query}")
            ]
        )

    def init_model(self):
        self.model = ChatOpenAI(
            api_key = self.openai_api_key,
            base_url = self.openai_api_base,
            model_name = self.model_name
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
            history_to_take = history[-history_sz:]
            history_to_take = [{'role': hist['role'], 'content': hist['content']} for hist in history_to_take]
            function_calls = self.chain_fn.invoke({"query": query, "history": history_to_take})
            for function_call in function_calls.split():
                if function_call == "Chain_Mongo":
                    logging.info(f'Calling Chain_Mongo with query: {query} and history: {history}')
                    return self.mongo_chain.call_chain(query, history)
                elif function_call == "Chain_General":
                    logging.info(f'Calling Chain_General with query: {query} and history: {history}')
                    return self.general_chain.call_chain(query, history)
                else:
                    logging.error(f'Unknown chain name: {function_call}. Or bad formatting of the output.')
                    query_new = f"Unknown chain name: {function_call}. Or bad formatting of the output."
                    return self.call_chain(query_new, history)
        except Exception as e:
            return str(e)
        
if __name__ == "__main__":
    import gradio as gr 
    router = ChainRouter()
    gr.ChatInterface(router.call_chain, type = "messages").launch(share = False)
    # assert router.call_chain("What is the capital of France?", []) == "Chain_General"
    # assert router.call_chain("Count the number of errors with destination as GNB?", []) == "Chain_Mongo"