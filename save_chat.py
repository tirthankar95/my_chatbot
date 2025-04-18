import logging 
from pymongo import MongoClient 
from time import time 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

class Save_Chat():
    def __init__(self, database_name: str = "SavedChats", collection_name: str = "Others"):
        ## MongoDB connection
        host, port = "localhost", 27017
        self.client = MongoClient(f"mongodb://{host}/{port}")
        self.collection = self.client[f"{database_name}"][f"{collection_name}"]
    def insert(self, history):
        hist_copy = history.copy()
        hist_copy["timestamp"] = time()
        self.collection.insert_one(hist_copy)
    def insert_many(self, history):
        hist_copy = history.copy()
        for convo in hist_copy:
            convo["timestamp"] = time()
        self.collection.insert_many(hist_copy)
        