from chain_base import Chains 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class Chain_Mongo(Chains):
    def __init__(self):
        super().__init__()
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