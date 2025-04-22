import re 
from chain_base import Chains, MIN_CHAT_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pymongo import MongoClient
from uuid import uuid4
from typing import List, Dict
from save_chat import Save_Chat
import logging 
import copy
from models import LM_Models
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

class Chain_Mongo(Chains):
    """Chain_Mongo class handles human queries related to data retrieval from MongoDB.
It is specifically designed to interpret and convert human questions into appropriate MongoDB queries.
Typical queries that involve keywords such as 'count', 'how many', 'errors', or similar fall under this class's scope.
"""
    def __init__(self, session_id: str, chroma_db_dir: str = "LLM_MONGO_2") -> None:
        super().__init__()
        ## Get Chat Object
        self.chat_obj_hidden = Save_Chat(collection_name = '[Train]')
        ## MongoDB connection
        host, port = "localhost", 27017
        self.client = MongoClient(f"mongodb://{host}:{port}/")
        self._collection = self.client["LLMQueryAgent"]["Functional"]
        ## Retriever.
        self.all_lm_models = LM_Models()
        self.chroma_db_dir = chroma_db_dir
        self.vector_store = Chroma(collection_name = "ONE_SHOT_EXAMPLES", \
                                   embedding_function = self.all_lm_models.embed_model, \
                                   persist_directory = self.chroma_db_dir)
        self.retriver = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        ## BUILD LangChain 
        self.prompt()
        self.chain_fn = self.__vector_retriver | self.prmpt | self.all_lm_models.lm_model | \
                        StrOutputParser() | self.__mongo_exec

    @property
    def collection(self):
        return self._collection
    @collection.setter
    def collection(self, value):
        '''
        Set the MongoDB collection to be used for the retriever, from the default one.
        The value should be a tuple of (db_name, collection_name).
        '''
        db_name, collection_name = value
        self._collection = self.client[db_name][collection_name]

    def __vector_retriver(self, input_2llm):
        query, history = input_2llm.get('query', None), input_2llm.get('history', None)
        rdocs = self.retriver.invoke(query)
        fmt_docx = "\n".join([doc.page_content for doc in rdocs])
        # Print LLM input. 
        logging.info(f'one_shot: {fmt_docx}\n')
        logging.info(f'history: {history}\n')
        logging.info(f'query: {query}\n')
        return {"schema": self.collection.find_one({}), \
                "one_shot": fmt_docx, \
                "history": history, \
                "query": query}

    def __mongo_exec(self, ai_message):
        try:
            logging.info(f'AI mongo query: {ai_message}')
            return f"""Executing AI mongo query: {ai_message} gives result:\n{eval(ai_message)}."""
        except Exception as e:
            logging.error(f"Error in executing mongo query: {ai_message} | {e}")
            raise Exception(f"{ai_message} | {e}")

    def prompt(self):
        self.prmpt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert AI assistant specializing in generating efficient and accurate MongoDB queries.""" + \
                   """A database entry is as follows: {schema}.\nHere is an example of mongo query interaction you must follow this assisstant response: {one_shot}.\n""" + \
                   """When returning a query, output only the query itself with no extra words or explanations or punctuations,""" + \
                   """so that an human can directly run the command. Donot put a fullstop at the end of the query."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}")
            ]
        )
    
    def call_chain(self, query: str, history: List[Dict]) -> str:
        '''
        This function calls the lang chain for a specific child class.
        Args:
            query (str): The human query that will be passed to the LLM chain.
            history List[Dict]: The chat history that will be passed to the LLM chain.
        Return:
            output (str): The output string from the LLM chain.
        '''
        logging.info('-m' * 30 + '\n')
        retry = MIN_CHAT_HISTORY
        resp = f"Retry limit exceeded."
        temp_hidden_history = []
        history_static = copy.deepcopy(history)
        while retry:
            history_to_take = history[-min(MIN_CHAT_HISTORY, len(history_static)):]
            try:
                for msg in self.prmpt.invoke(self.__vector_retriver({"query": query, \
                                                                     "history": history_to_take})).to_messages():
                    temp_hidden_history.append({
                        'role': msg.type, 
                        'content': msg.content
                        })
                    history_static.append({
                        'role': msg.type, 
                        'content': msg.content
                        })
                resp = self.chain_fn.invoke({"query": query, "history": history_to_take})
                retry = 0
            except Exception as ai_message_error:
                ai_message, error = str(ai_message_error).split('|')
                '''
                Manually adding histories for re-tries. 
                This won't affect the main history which originates at the router. 
                '''
                temp_hidden_history.append({
                    "role": "ai", 
                    "content": f"{ai_message.strip()}"
                    })
                history_static.append({
                    "role": "ai", 
                    "content": f"{ai_message.strip()}"
                    })
                query = f"Error: {error.strip()}.\nPlease rephrase your mongo query."
                retry -= 1
        temp_hidden_history.append({
                "role": "ai", 
                "content": f"{resp.strip()}"
            })
        self.chat_obj_hidden.insert_serialize(temp_hidden_history)
        return resp
    
    def add_one_shots(self):
        examples = [
            Document(
                page_content = """human: Count the number of errors with source as UDR?\n ai:self.collection.count_documents({"src": "UDR"})""",
                metadata = {"source": "manual_tmittra"}),
            Document(
                page_content = """human: Display errors with destination as UDR?\nai: list(self.collection.find({"dst": "UDR"}))""",
                metadata = {"source": "manual_tmittra"})
            ]
        uuids = [str(uuid4()) for _ in range(len(examples))]
        self.vector_store.add_documents(documents = examples, ids = uuids)

    def exec_query(self, query: str) -> str:
        '''
        This function executes a query on the MongoDB collection.
        Args:
            query (str): The query to be executed.
        Return:
            void
        '''
        try:
            logging.info(eval(query))
        except Exception as e:
            logging.info(str(e))

if __name__ == "__main__":
    import gradio as gr 
    mongo_chain = Chain_Mongo(session_id="OneShot")
    mongo_chain.add_one_shots()
    gr.ChatInterface(mongo_chain.call_chain, type = "messages").launch(share = False)
    # Query 1: Count the number of errors with destination as GNB?
    # Query 2: Display errors with destination as UDM?
    # mongo_chain.exec_query("""list(self.collection.find({"dst": "UDM", "Error_Markers.type": "http2"}))""")