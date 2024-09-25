from fastapi import FastAPI, HTTPException
from logger import logger
import httpx
import json
import re
import os
import openai
from typing import List, Dict
import ast



# Load your OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

async def transform_query_chatbot(conversation: list[dict]):
        try:
            logger.info(f"Received request to transform query: {conversation}")

            # Retrieve API key
            gpt_api_key = os.getenv("OPENAI_API_KEY")
            if not gpt_api_key:
                logger.error("OpenAI API key not found")
                raise HTTPException(status_code=500, detail="OpenAI API key not found")
            
            # messages = [
            #     #{"role": "system", "content": "You are an assistant that converts natural language into MongoDB queries in python, please give me only the code."},
            #     {"role": "system", "content": INSTRUCTIONS1 + MONGO_COLLECTION_FIELDS + INSTRUCTIONS2},
            #     {"role": "user", "content": f"Transform the following natural language request into a MongoDB query in pymongo:\n\"{user_query}\".format()"}
            # ]

            headers = {
                "Authorization": f"Bearer {gpt_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "gpt-3.5-turbo-16k",
                # "model": "gpt-4o-2024-08-06",
                "messages": conversation,
                "max_tokens": 100,
                "temperature": 0.5,
            }

            logger.info("Sending request to OpenAI API...")
            async with httpx.AsyncClient() as client:
                response = await client.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)

                logger.debug(f"OpenAI API response status code: {response.status_code}")
                if response.status_code != 200:
                    logger.error(f"Failed to transform query: {response.text}")
                    raise HTTPException(status_code=500, detail="Error transforming query")

                response_json = response.json()
                logger.info(f"OpenAI API response: {response_json}")

                mongo_query = response_json["choices"][0]["message"]["content"].strip()
                # Convert the MongoDB query string into a dictionary
                logger.info("loads data from DB")
                # logger.info("Raw MongoDB Query: ", mongo_query)
                print("Raw MongoDB Query: ", mongo_query)

                # Check for irrelevant input message
                if "There is no relevant query for the given natural language request" in mongo_query:
                    return mongo_query

                # Extract the MongoDB query part from the response
                match = re.search(r"query\s*=\s*({.*?})", mongo_query, re.DOTALL)
                print("match: ", match)
                if not match:
                    #match = re.search(r"(\{.*?\})", merged_query, re.DOTALL)  # Match the dictionary format
                    #return json.loads(match.group(1))
                    return ast.literal_eval(mongo_query)
                if match:
                    query_str = match.group(1)  # Extract the content inside the braces

                    # Replace any variable (like actor_name) with its value
                    variable_assignments = re.findall(r"(\w+)\s*=\s*\"(.*?)\"", mongo_query)
                    for var, value in variable_assignments:
                        query_str = query_str.replace(var, f'"{value}"')

                    print(f'{query_str=}')
                    mongo_query_dict = eval(query_str)  # Convert to dictionary
                    logger.info(f"Generated MongoDB query: {mongo_query_dict}")
                    return mongo_query_dict
                else:
                    logger.info("Failed to extract MongoDB query from the response.")
                    raise HTTPException(status_code=500, detail="Failed to extract MongoDB query.")


        except Exception as e:
            logger.info(f"Error transforming query: {e}")
            raise HTTPException(status_code=500, detail="Error transforming query")
        




# Function to generate a natural language response based on the best movie match
async def generate_natural_response(best_movie: dict) -> str:
    """
    Generates a natural language response describing the best movie match. Always start with the movie name, locate at "movie_title" field im mongoDB.
    
    Args:
    - best_movie (dict): A dictionary containing details of the best-matched movie.
    
    Returns:
    - str: A natural language response summarizing the movie.
    """
    try:
        movie_name = best_movie.get("movie_title", "Unknown Movie")
        actors = ', '.join(best_movie.get("actors", []))
        rank = best_movie.get("rank", "Unknown Rank")
        description = best_movie.get("description", "No description available.")

        # Prepare the prompt for the LLM
        movie_summary = (f"Movie {movie_name} starring {actors} is ranked {rank} and is about {description}. "
                         "Please summarize this in a more user-friendly way.")

        # Generate the response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": movie_summary}
            ]
        )

        # Extract the generated text from OpenAI's response
        #natural_response = response.choices[0].message['content']
        natural_response = response['choices'][0]['message']['content']
        return natural_response

    except Exception as e:
        # Handle any error and return a default message
        return f"Error generating natural response: {str(e)}"

