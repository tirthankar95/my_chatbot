import os 
import dotenv
import logging 
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download, list_repo_files
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
dotenv.load_dotenv()

class LM_Models():
    # Singleton Instance
    _instance = None 
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance 
    
    def __init__(self, model_name: str = "qwen2.5-7b-instruct-q4_0.gguf", \
                 embed_model: str = "thenlper/gte-base", embed_model_dir: str = "./embed_model/"):
        # Prevent reinitialization if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        # LANGUAGE MODEL
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = "http://localhost:8000/v1" if "USE_OPENAI" not in os.environ else None 
        self.model_name = model_name if "MODEL_NAME" not in os.environ else os.getenv("MODEL_NAME")
        logging.info(f'Model[{self.model_name}], Base_URL[{self.openai_api_base}]')
        self._lm_model = ChatOpenAI(
            api_key = self.openai_api_key,
            model_name = self.model_name
        )
        # EMBED MODEL
        self.local_dir = embed_model_dir
        self.repo_id = embed_model
        if not os.path.exists(self.local_dir):
            filenames = list_repo_files(self.repo_id)
            for file in filenames:
                if not os.path.exists(os.path.join(self.local_dir, file)):
                    hf_hub_download(self.repo_id, file, local_dir = self.local_dir)
        self.embed_model = HuggingFaceEmbeddings(model_name = self.repo_id, \
                                                 model_kwargs = {'device': 'cpu'}, \
                                                 encode_kwargs = {'normalize_embeddings': True})
        
    @property
    def lm_model(self):
        return self._lm_model
    @lm_model.setter
    def lm_model(self, ChatModel):
        self._lm_model = ChatModel

    @property
    def embed_model(self):
        return self._embed_model
    @embed_model.setter
    def embed_model(self, EmbedModel):
        self._embed_model = EmbedModel
