import os
import openai
import base64
import requests
import re
import ast
# Function to encode the image
rlbench_env = "/home/duanj1/m2t2/manipulate_anything/RLBench"
qwen_env = "/home/duanj1/CameraCalibration/"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_with_highest_number(folder_path):
    # Initialize variables
    highest_number = -1
    image_with_highest_number = None
    
    # Regex to find numbers in filenames
    number_pattern = re.compile(r'(\d+)')

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Extract the number from the filename
        match = number_pattern.search(filename)
        if match:
            number = int(match.group(0))
            # Update the highest number and corresponding image if necessary
            if number > highest_number:
                highest_number = number
                image_with_highest_number = filename
    
    if image_with_highest_number:
        return os.path.join(folder_path, image_with_highest_number)
    else:
        return None

# Define your action and object
file_path = rlbench_env+'/action_info.txt'
with open(file_path, 'r') as file:
    content = file.read()
    data_dict = ast.literal_eval(content)
next_action = data_dict['primitive_actions'][0]
next_obj =  data_dict['Objects'][0]


action = str(next_action)
object = str(next_obj)

# Paths
folder_path = rlbench_env+'/save_frames/'
text_file_path = qwen_env+"LLMs/LLM_Progprompt/create_action.txt"
output_file_path = rlbench_env+'/action_code.txt'

# Get the latest image
highest_image_path = get_image_with_highest_number(folder_path)
image_path = str(highest_image_path)
# print(f"Using image: {image_path}")

# Encode the image
base64_image = encode_image(image_path)

# Read and format the text prompt
with open(text_file_path, 'r') as file:
    text_prompt_template = file.read().strip()

text_prompt = text_prompt_template.format(action=action, object=object)
# print(f"Formatted text prompt:\n{text_prompt}")
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
# print(output)

output_file_path = rlbench_env+'/action_code.txt'

# Clear the content of the output file
with open(output_file_path, 'w') as file:
    pass  # Opening in 'w' mode and closing will clear the file
# Now, write the new content
with open(output_file_path, 'w') as file:
    file.write(str(output))
