import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


chat_completion = client.chat.completions.create(
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
    