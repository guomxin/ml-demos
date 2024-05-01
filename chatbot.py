# Use a pipeline as a high-level helper
from transformers import pipeline

#pipe = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")
chatbot = pipeline("conversational", model="./models/facebook/blenderbot-400M-distill")

user_message = """
What are some fun activities I can do in the winter?
"""
from transformers import Conversation
conversation = Conversation(user_message)
conversation = chatbot(conversation)
print(conversation)

conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })
conversation = chatbot(conversation)
print(conversation)
