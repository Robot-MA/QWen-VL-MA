import os
import openai
import base64
import requests

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/table_item.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

# Read the text prompt from a file
text_file_path = "/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/create_task_prompt.txt"  # Replace with your text file path
with open(text_file_path, 'r') as file:
    text_prompt = file.read().strip()
print(text_prompt)
# Setting up headers and payload
api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 4000
}

# Making the request
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Print the response
output= response.json()['choices'][0]['message']['content']
print(output)

output_file_path = '/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/output_action.txt'

# Clear the content of the output file
with open(output_file_path, 'w') as file:
    pass  # Opening in 'w' mode and closing will clear the file

# Now, write the new content
with open(output_file_path, 'w') as file:
    file.write(str(output))

