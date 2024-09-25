# app/database/conversation_model.py

from app.database.db_connection import db
from app.database.db_models import Conversation, Message
from pymongo.collection import Collection
from datetime import datetime

conversation_collection: Collection = db["Conversations"]

def save_conversation(conversation_id: str, user_message: str, system_response: str):
    # Fetch the existing conversation
    conversation = conversation_collection.find_one({"conversation_id": conversation_id})

    new_message_user = {"role": "user", "content": user_message, "timestamp": datetime.now()}
    new_message_system = {"role": "system", "content": system_response, "timestamp": datetime.now()}

    if conversation:
        # Update the conversation by appending new messages and limiting to the last 5 exchanges
        messages = conversation["messages"][-8:] + [new_message_user, new_message_system]
        print("messages",messages)

        conversation_collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"messages": messages, "updated_at": datetime.now()}}
        )

    else:
        # Create a new conversation
        print(conversation_id, new_message_user, new_message_system)
        # new_conversation = Conversation(
        #     conversation_id=conversation_id,
        #     messages=[new_message_user, new_message_system]
        # )
        conversation_collection.insert_one(
            {
                "conversation_id": conversation_id,
                "messages": [new_message_user, new_message_system],
            }
        )

def get_conversation(conversation_id: str, limit: int = 5):
    print("in get_convers", conversation_id)

    # Fetch the conversation data from the database
    conversation_data = list(conversation_collection.find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit))

    print("conversation_data", conversation_data)
    
    # Filter out documents that don't have 'role' or 'content'
    valid_messages = []

    for conversation in conversation_data:
        messages = conversation.get('messages', [])
        valid_messages.extend([message for message in messages if 'role' in message and 'content' in message])
    
    # Create Conversation object with valid messages
    print("123")
    conversation = Conversation(
        conversation_id=conversation_id,
        messages=[Message(**message) for message in valid_messages]  # Unpack only valid messages
    )
    
    print("conversation", conversation)

    # Iterate over the conversation messages and extract desired fields
    conversation_history = [{"role": message.role, "content": message.content} for message in conversation.messages]
    print("conversation_history", conversation_history)

    return conversation_history  # Or return the whole Conversation object if needed

