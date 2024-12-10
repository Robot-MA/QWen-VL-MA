import os
import openai
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key 
# print(openai.Model.list())

def get_completion(prompt, model="gpt-4"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(

    model=model,

    messages=messages,

    temperature=0,

    )

    return response.choices[0].message["content"]

prompt = "Hi"

response = get_completion(prompt)

print(response)