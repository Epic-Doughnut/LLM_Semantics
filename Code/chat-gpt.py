import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

chat_completion = openai.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

# def get_completion(prompt, model="gpt-3.5-turbo"):
# 	messages = [{"role": "user", "content": prompt}]

# 	response = openai.ChatCompletion.create(	model=model,	messages=messages,	temperature=0,	)
# 	return response.choices[0].message["content"]

if __name__ == "__main__":
    
    print(chat_completion)
    