import re 
from chain_base import Chains 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class Chain_Mongo(Chains):
    def __init__(self):
        super().__init__()
    
    def __mongo_exec(self, ai_message):
        try:
            print(f'AI mongo query: {ai_message}')
            return f"The result of the query is: {eval(ai_message)}."
        except Exception as e:
            pattern = re.compile(r"\bcollection")
            if pattern.search(ai_message):
                return f"Extract only the query from {ai_message}."
            else:
                return f"Invalid query. Please try again."
    
    def prompt(self):
        self.prmpt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert AI assistant specializing in generating efficient and accurate MongoDB queries.""" + \
                   """A database entry is as follows: {schema}.\nHere is an example of mongo query interaction you must follow this assisstant response: {one_shot}.\n""" + \
                   """When returning a query, output only the query itself with no extra words or explanations or punctuations,""" + \
                   """so that an user can directly run the command."""),
                   MessagesPlaceholder(variable_name="history"),
                   ("user", "{query}")
            ]
        )