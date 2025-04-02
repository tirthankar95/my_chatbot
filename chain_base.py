from abc import abstractmethod, ABC 

class Chains(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def prompt(self) -> None:
        '''
        This function denotes the prompt specific to a langchain
        Args:
            None
        Return:
            None
        '''
        pass 

    @abstractmethod
    def init_model(self) -> None:
        '''
        This function denotes the LLM models for each langchain
        Args:
            None
        Return:
            None 
        '''
        pass
    