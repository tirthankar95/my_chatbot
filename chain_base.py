from abc import abstractmethod, ABC 

class Chains(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def prompt(self):
        '''
        '''
    @abstractmethod
    def init_model(self):
        '''
        '''
    