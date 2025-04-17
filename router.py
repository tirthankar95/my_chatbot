from chain_base import MIN_CHAT_HISTORY
from chain_mongo import Chain_Mongo
from chain_general import Chain_General
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from uuid import uuid4
from models import LM_Models
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
from typing import List, Dict
from time import time 
from save_chat import Save_Chat

class ChainRouter():
    def __init__(self) -> None:
        super().__init__()
        ## Save Chat
        self.session_id = uuid4().hex[:16]
        self.chat_obj = Save_Chat(collection_name = self.session_id)
        ## Init LM models & prompt
        self.all_lm_models = LM_Models()
        self.prompt()
        ## Init Chains
        self.mongo_chain = Chain_Mongo(session_id = self.session_id)
        self.general_chain = Chain_General(session_id = self.session_id)
        self.chain_fn = self.prmpt | self.all_lm_models.lm_model | StrOutputParser()
        self.history = []

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
    
    def call_chain(self, query: str) -> str:
        '''
        This function calls the lang chain for a specific child class.
        Args:
            query (str): The user query that will be passed to the LLM chain.
        Return:
            output (str): The output string from the LLM chain.
        '''
        logging.info('-r' * 30 + '\n')
        history_sz = min(MIN_CHAT_HISTORY, len(self.history))
        chat_session, resp, retry = [], "", 2
        function_calls = []
        # Start planning chain.
        history_to_take = self.history[-history_sz:]
        query_planner = query
        ## Validate if the name of the function is correct.
        while retry > 0:
            function_calls = self.chain_fn.invoke({"query": query_planner, "history": history_to_take})
            query_planner = ""
            for function_call in function_calls.split():
                if function_call not in ["Chain_Mongo", "Chain_General"]:
                    logging.error(f'Unknown chain name: {function_call} or bad formatting of the output with ' + \
                                f'query: {query} and history: {self.history}')
                    query_planner += f"Unknown chain name: {function_call} or bad formatting of the output."
            if len(query_planner) == 0: break                
            retry -= 1
        ## Insert planner query-response from LLM.
        self.history.extend([
            {                
                'role': 'user',
                'content': query
            },
            {
                'role': 'assistant',
                'content': function_calls
            }])
        chat_session.extend([
            {
                "timestamp": time(),
                "role": "user", 
                "content": query, 
                "content_train": self.prmpt.invoke({"query": query, "history": history_to_take}).to_string()
            },
            {
                "timestamp": time(),
                "role": "assistant",
                "content": function_calls,
            }])
        for function_call in function_calls.split():
            if function_call == "Chain_Mongo":
                logging.info(f'Calling Chain_Mongo with query: {query} and history: {self.history}')
                # Each mongo query session is independent, so no need to pass the history in the session.
                resp = self.mongo_chain.call_chain(query, [])
            elif function_call == "Chain_General":
                logging.info(f'Calling Chain_General with query: {query} and history: {self.history}')
                resp = self.general_chain.call_chain(query, self.history)
        logging.info(f'Chat Insertion: Router')
        self.chat_obj.insert(chat_session)
        return resp
        
    def call_chain_gr(self, query: str, history: List[Dict]) -> str:
        '''
        This function calls the lang chain for a specific child class.
        Args:
            query (str): The user query that will be passed to the LLM chain.
            history List[Dict]: The chat history that will be passed to the LLM chain.
        Return:
            output (str): The output string from the LLM chain.
        '''
        logging.info('-r' * 30 + '\n')
        history_sz = min(MIN_CHAT_HISTORY, len(history))
        chat_session, resp, retry = [], "", 2
        function_calls = []
        try:
            history_to_take = history[-history_sz:]
            history_to_take = [{'role': hist['role'], 'content': hist['content']} for hist in history_to_take]
            query_planner = query
            ## Validate if the name of the function is correct. ##
            while retry > 0:
                function_calls = self.chain_fn.invoke({"query": query_planner, "history": history_to_take})
                query_planner = ""
                for function_call in function_calls.split():
                    if function_call not in ["Chain_Mongo", "Chain_General"]:
                        logging.error(f'Unknown chain name: {function_call} or bad formatting of the output with ' + \
                                    f'query: {query} and history: {history}')
                        query_planner += f"Unknown chain name: {function_call} or bad formatting of the output."
                if len(query_planner) == 0: break                
                retry -= 1
            ## Insert planner query-response from LLM.
            chat_session.extend([
                {
                    "timestamp": time(),
                    "role": "user", 
                    "content": query, 
                    "content_train": self.prmpt.invoke({"query": query, "history": history_to_take}).to_string()
                },
                {
                    "timestamp": time(),
                    "role": "assistant",
                    "content": function_calls,
                }])
            for function_call in function_calls.split():
                if function_call == "Chain_Mongo":
                    logging.info(f'Calling Chain_Mongo with query: {query} and history: {history}')
                    # Each mongo query session is independent, so no need to pass the history in the session.
                    resp = self.mongo_chain.call_chain(query, [])
                elif function_call == "Chain_General":
                    logging.info(f'Calling Chain_General with query: {query} and history: {history}')
                    resp = self.general_chain.call_chain(query, history)
        except Exception as e:
            resp = str(e)
        finally:
            logging.info(f'Chat Insertion: Router')
            self.chat_obj.insert(chat_session)
            return resp
            
        
if __name__ == "__main__":
    import gradio as gr 
    router = ChainRouter()
    gr.ChatInterface(router.call_chain_gr, type = "messages").launch(share = False)
    # assert router.call_chain("What is the capital of France?", []) == "Chain_General"
    # assert router.call_chain("Count the number of errors with destination as GNB?", []) == "Chain_Mongo"