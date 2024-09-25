
# #-----------------------------2 embedding fields----------------------------#

# import csv
# from typing import List
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from db import movies_collection  # Import from your existing db.py

# # Initialize SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def load_movie_data(file_path: str) -> List[dict]:
#     """Load movie data from a CSV file."""
#     movies_data = []
#     with open(file_path, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             movies_data.append(row)
#     return movies_data

# def generate_embeddings(text: str) -> List[float]:
#     """Generate vector embeddings for a given text."""
#     return model.encode(text).tolist()

# def add_embeddings_to_movies(file_path: str) -> None:
#     """Add vector embeddings to the Movies collection."""
#     movies_data = load_movie_data(file_path)
#     for movie in movies_data:
#         print(movie.keys())  # Add this line to debug the keys
#         description = movie.get('description', '')
#         critics_consensus = movie.get('critics_consensus', '')
        
#         # Generate embeddings for both fields
#         description_embedding = generate_embeddings(description)
#         critics_consensus_embedding = generate_embeddings(critics_consensus)
        
#         # Update the MongoDB collection with both embeddings
#         movies_collection.update_one(
#             {'movie_title': movie['movie_title']},  # Assuming 'movie_title' is the unique identifier
#             {'$set': {
#                 'description_embedding': description_embedding,
#                 'critics_consensus_embedding': critics_consensus_embedding
#             }}
#         )
#     print("Embeddings added to the Movies collection successfully!")

# # Example usage
# add_embeddings_to_movies('data/Movies_DB_2.json')


import json
from typing import List
from sentence_transformers import SentenceTransformer
from db import movies_collection  # Import from your existing db.py

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_movie_data(file_path: str) -> List[dict]:
    """Load movie data from a JSON file."""
    with open(file_path, 'r') as f:
        movies_data = json.load(f)
    return movies_data

def generate_embeddings(text: str) -> List[float]:
    """Generate vector embeddings for a given text."""
    return model.encode(text).tolist()

def add_embeddings_to_movies(file_path: str) -> None:
    """Add vector embeddings to the Movies collection."""
    movies_data = load_movie_data(file_path)
    for movie in movies_data:
        print(movie.keys())  # Debugging: Print the keys of each movie dictionary
        description = movie.get('description', '')
        critics_consensus = movie.get('critics_consensus', '')
        
        # Generate embeddings for both fields
        description_embedding = generate_embeddings(description)
        critics_consensus_embedding = generate_embeddings(critics_consensus)
        
        # Update the MongoDB collection with both embeddings
        movies_collection.update_one(
            {'movie_title': movie['movie_title']},  # Assuming 'movie_title' is the unique identifier
            {'$set': {
                'description_embedding': description_embedding,
                'critics_consensus_embedding': critics_consensus_embedding
            }}
        )
    print("Embeddings added to the Movies collection successfully!")

# Example usage
add_embeddings_to_movies('data/Movies_DB_2.json')
