import re 
from chain_base import Chains, MIN_CHAT_HISTORY
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download, list_repo_files
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pymongo import MongoClient
import os 
import dotenv
from uuid import uuid4
from typing import List, Dict
import logging 
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()
class Chain_Mongo(Chains):
    """class Chain_Mongo: is used to answer user queries with MongoDB."""
    def __init__(self, model_name: str = "qwen2.5-7b-instruct-q4_0.gguf", \
                 embed_model: str = "thenlper/gte-base", embed_model_dir: str = "./embed_model/", \
                 chroma_db_dir: str = "LLM_MONGO_1") -> None:
        super().__init__()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_api_base = "http://localhost:8000/v1"
        self.model_name = model_name
        self.local_dir = embed_model_dir
        self.repo_id = embed_model
        ## MongoDB connection
        host, port = "localhost", 27017
        self.client = MongoClient(f"mongodb://{host}:{port}/")
        self._collection = self.client["LLMQueryAgent"]["Functional"]
        ## BUILD LangChain 
        self.init_model()
        # Retriever.
        self.chroma_db_dir = chroma_db_dir
        self.vector_store = Chroma(collection_name = "ONE_SHOT_EXAMPLES", \
                                   embedding_function = self.embed_model, \
                                   persist_directory = self.chroma_db_dir)
        self.retriver = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        # Get Prompt.
        self.prompt()
        # Build chain.
        self.chain_fn = self.__vector_retriver | self.prmpt | self.model | \
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
        query, history = input_2llm['query'], input_2llm['history']
        rdocs = self.retriver.invoke(query)
        fmt_docx = "\n".join([doc.page_content for doc in rdocs])
        # Print LLM input. 
        logging.info('--' * 50 + '\n')
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
            return f"""Executing AI mongo query: {ai_message} gives result:\n {eval(ai_message)}."""
        except Exception as e:
            raise Exception(f"{ai_message} | {e}")
    
    def init_model(self):
        self.model = ChatOpenAI(
            api_key = self.openai_api_key,
            base_url = self.openai_api_base,
            model_name = self.model_name
        )
        filenames = list_repo_files(self.repo_id)
        for file in filenames:
            if not os.path.exists(os.path.join(self.local_dir, file)):
                hf_hub_download(self.repo_id, file, local_dir = self.local_dir)
        # Load embedding model.
        self.embed_model = HuggingFaceEmbeddings(model_name = self.repo_id, \
                                model_kwargs = {'device': 'cpu'}, \
                                encode_kwargs = {'normalize_embeddings': True})

    def prompt(self):
        self.prmpt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert AI assistant specializing in generating efficient and accurate MongoDB queries.""" + \
                   """A database entry is as follows: {schema}.\nHere is an example of mongo query interaction you must follow this assisstant response: {one_shot}.\n""" + \
                   """When returning a query, output only the query itself with no extra words or explanations or punctuations,""" + \
                   """so that an user can directly run the command. Donot put a fullstop at the end of the query."""),
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
        retry, curr_query = MIN_CHAT_HISTORY, query 
        while retry:
            try:
                return self.chain_fn.invoke({"query": curr_query, "history": history})
            except Exception as ai_message_error:
                prev_query = curr_query
                ai_message, error = str(ai_message_error).split('|')
                '''
                Manually adding histories for re-tries. 
                This won't affect the main history which originates at the router. 
                '''
                history.append({"role": "user", "content": f"{prev_query}"})
                history.append({"role": "assistant", "content": f"{ai_message.strip()}"})
                curr_query = f"Error: {error.strip()}.\nPlease rephrase your query."
                retry -= 1
        return f"Retry limit reached. Query:\n{query}."
    
    def add_one_shots(self):
        examples = [
            Document(
                page_content = """user: Count the number of errors with source as UDR?\n assistant:self.collection.count_documents({"src": "UDR"})""",
                metadata = {"source": "manual_tmittra"}),
            Document(
                page_content = """user: Display errors with destination as UDR?\nassistant: list(self.collection.find({"dst": "UDR"}))""",
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
    mongo_chain = Chain_Mongo()
    mongo_chain.add_one_shots()
    gr.ChatInterface(mongo_chain.call_chain, type = "messages").launch(share = False)
    # Query 1: Count the number of errors with destination as GNB?
    # Query 2: Display errors with destination as UDM?
    # mongo_chain.exec_query("""list(self.collection.find({"dst": "UDM", "Error_Markers.type": "http2"}))""")