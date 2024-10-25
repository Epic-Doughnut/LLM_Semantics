# import os
# import openai
from dotenv import load_dotenv

# # Load environment variables from .env file
load_dotenv()

# # Load OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the GPT-4 model for chat completion.
# GPT_MODEL = "gpt-4o-mini"
from openai import OpenAI 
import os

## Set the API key and model name
MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
    {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
  ]
)

print("Assistant: " + completion.choices[0].message.content)