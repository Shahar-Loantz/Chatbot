from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve MongoDB URI from environment variable
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["Movies_DB_2"]
movies_collection = db["Movies"]
actors_collection = db["Actors"]
print("DB connected")