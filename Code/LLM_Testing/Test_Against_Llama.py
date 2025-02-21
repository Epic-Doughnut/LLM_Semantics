from groq import Groq

class Llama_Test:
    def Recommend_for_Sample(sample_file, output_file):
        with open(sample_file) as inf:
            descriptions = inf.read()
        with open("API_Key.txt") as api_file:
            key = api_file.read()
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Recommend me another book that would be suitable for a reader of the same age as the one that read a book with each of the following descriptions:" + descriptions
                },
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        with open(output_file, 'w') as of:
            of.write(completion.choices[0].message.content)
    

Llama_Test.Recommend_for_Sample("Sample_Data.txt", "LLM_Recommendations.txt")

