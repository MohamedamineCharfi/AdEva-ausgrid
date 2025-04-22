from pymongo import MongoClient
from django.conf import settings

# MongoDB Configuration
MONGO_DB = {
    'host': 'mongodb://localhost:27017',
    'name': 'energy_db'
}

client = MongoClient(MONGO_DB['host'])
db = client[MONGO_DB['name']]

def get_mongo_collection(collection_name):
    return db[collection_name]