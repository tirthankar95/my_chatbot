from abc import abstractmethod, ABC 
from typing import List, Dict
MIN_CHAT_HISTORY = 5
class Chains(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def prompt(self) -> None:
        '''
        This function defines the prompt specific to a langchain.
        Args:
            None
        Return:
            None
        '''
        pass 

    @abstractmethod
    def call_chain(self, query: str, history: List[Dict]) -> str:
        '''
        This function calls the lang chain for a specific child class.
        Args:
            query (str): The user query that will be passed to the LLM chain.
            history List[Dict]: The chat history that will be passed to the LLM chain.
        Return:
            output (str): The output string from the LLM chain.
        '''
        pass