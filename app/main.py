from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from app.openai_api import transform_query_chatbot, generate_natural_response
from app.database.db_connection import movies_collection as db
from pydantic import BaseModel
from logger import logger
import numpy as np
from app.database.conversation_model import save_conversation, get_conversation
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


# Allow all origins for development; adjust as needed for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the SentenceTransformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device="cpu")

INSTRUCTIONS1 = (
    "You are an assistant that converts natural language into MongoDB queries. "
    "The database contains a collection of movies with the following fields:\n"
)
INSTRUCTIONS2 = (
    "Your task is to generate a MongoDB query in Python's pymongo format "
    "based on a given natural language request. Ensure that the query only returns relevant documents. "
    "The response structure should look like: query={'field': 'value'}. For example: {extracted_genres: 'Drama', extracted_audience: 'Adults'}. "
    "If the request does not relate to any of the available movie fields, respond with the string: "
    "'There is no relevant query for the given natural language request in the context of the movie database.'"
)

MONGO_COLLECTION_FIELDS = (
    "- _id: ObjectId (e.g., 66c1e56f2df6ccb5696503c0)\n"
    "- movie_title: String of movie name (e.g., 'Percy Jackson & the Olympians: The Lightning Thief')\n"
    "- rotten_tomatoes_link: String (e.g., 'm/0814255')\n"
    "- description: String (a brief summary of the movie)\n"
    "- critics_consensus: String (a summary of critics' opinions)\n"
    "- extracted_genres: Array of Strings (e.g., ['Action', 'Drama'])\n"
    "- actors: Array of Strings (e.g., ['Logan Lerman', 'Pierce Brosnan'])\n"
    "- runtime: Int32 (the runtime of the movie in minutes)\n"
    "- rank: Int32 (the rank of the movie)\n"
    "- created_at: String (timestamp when the document was created)\n"
    "- updated_at: String (timestamp when the document was last updated)\n\n"
    "- description_embedding: Array of Strings"
)


class QueryRequest(BaseModel):
    conversation_id: str
    user_query: str

# Function to compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Unified endpoint to handle movie search, similarity scoring, and conversation management
@app.post("/search-movies/")
async def search_and_chat(request: QueryRequest):
    conversation_id = request.conversation_id
    user_query = request.user_query
    logger.info(f"User query: {user_query} for user ID: {conversation_id}")

    try:
        # Step 1: Retrieve the conversation history (limit to last 5 messages)
        print("trying to get in conversation")
        conversation = get_conversation(conversation_id, limit=5)
        print("after get_conversation! ",conversation)

        #conversation_history = [{"role": message["role"], "content": message["content"]} for message in conversation.get("messages", [])]
        conversation_history = [{"role": message["role"], "content": message["content"]} for message in conversation]
        print("conversation_history: ",conversation_history)
        conversations = [{"role": "system", "content": INSTRUCTIONS1 + MONGO_COLLECTION_FIELDS + INSTRUCTIONS2}] + conversation_history
        if not conversation_history:
            conversations.append({"role": "user", "content": f"Transform the following natural language request into a MongoDB query in pymongo:\n\"{user_query}\""})
        # Step 2: Add the new user message to the conversation history
        conversations.append({"role": "user", "content": f"Refer to the recent responses and add the following to the pymongo query:\n\"{user_query}\""})
        print("conversations: ",conversations)

        # Step 3: Transform the user's query into MongoDB key-value pairs using OpenAI
        mongo_query_dict = await transform_query_chatbot(conversations)
        print("mongo_query_dict: ", mongo_query_dict)
        logger.info("MongoDB query dict: ", mongo_query_dict)
        print("mongo_query_dict from main: ", mongo_query_dict)

        if isinstance(mongo_query_dict, str):
            # If the result is a string, return the message
            return {"message": mongo_query_dict}

        # Modify query to handle partial matches and case insensitivity
        modified_query = {}
        for field, value in mongo_query_dict.items():
            if isinstance(value, int) or value.isdigit():
                modified_query[field] = int(value)
            else:
                modified_query[field] = {'$regex': f'.*{value}.*', '$options': 'i'}

        logger.info(f"Generated MongoDB query: {modified_query}")
        print("modified_query: ",modified_query)

        # Step 4: Fetch relevant movies from MongoDB collection
        movies = list(db.find(mongo_query_dict))

        if not movies:
            return {"message": "No movies found matching your query."}

        logger.info(f"Retrieved {len(movies)} movies from MongoDB.")

        # Step 5: Generate embedding for the user query
        query_embedding = model.encode(user_query, convert_to_tensor=True)
       
        # Step 6: Compare the query embedding with movie description embeddings
        best_match = None
        best_score = -1  # Initialize with a low score

        counter = 0
        for movie in movies:
            # Fetch the movie's description embedding
            movie_embedding = np.array(movie.get("description_embedding"))
            counter = counter + 1
            if movie_embedding is not None:
                # Calculate cosine similarity
                similarity_score = cosine_similarity(query_embedding, movie_embedding)
                movie["similarity_score"] = similarity_score

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = movie
        
        print(f'{best_match=}')
        if best_match:
            best_match["_id"] = str(best_match["_id"])  # Convert ObjectId to string

            # Step 7: Generate a natural language response based on the best match
            natural_response = await generate_natural_response(best_match)

            # Replace "Unknown" with the actual movie title
            movie_title = best_match.get("movie_title", "Unknown")
            natural_response = natural_response.replace("Unknown", movie_title)
            
            # Save the conversation (limit the conversation to the last 5 exchanges)
            save_conversation(conversation_id, user_query, natural_response)

            # Return the natural language response along with the best match details
            return {
                "natural_response": natural_response,
                "similarity_score": best_score,
                "best_match": best_match,
                "conversation_history": conversation_history  # Optionally return conversation history
            }
        else:
            return {"message": "No similar movie found."}

    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing the movie search request.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)



