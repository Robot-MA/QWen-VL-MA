import os
import base64
import requests
import json
import glob
from io import BytesIO
import tiktoken

# Get OpenAI API Key from environment variable
api_key = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

metaprompt = '''
 - You are looking at a list of images representing a single scene from different views. Your goal 
is to answer the question related to this scene correctly based on these list of images. 
Note that you should learn to build a comprehensive understanding of the whole 3D scene based on these images. You are an expert in this and you always do it well. 
The input format is like this: List of images, one question. 
And your answer should be in this format:
Answer: the answer to the question. Try to answer the question with one or two words. 
Here is one example:
<List of images> Question: What is the color of the table? Answer: Brown
'''    

all_images = {}
# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    import pdb;pdb.set_trace()
    for message in messages['messages']:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def prepare_inputs(message, image_paths, scene_id):

    # # Path to your image
    # image_path = "temp.jpg"
    # # Getting the base64 string
    # base64_image = encode_image(image_path)
    images = []

    if scene_id not in all_images.keys():
        for image in image_paths:
            b64_image = encode_image_from_file(image)
            images.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64_image}",
            })
        all_images[scene_id] = images
    else:
        images = all_images[scene_id]

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "system",
            "content": [
                metaprompt
            ]
        }, 
        {
            "role": "user",
            "content": images + [
            {
                "type": "text",
                "text": message, 
            }
            ]
        }
        ],
        "max_tokens": 800
    }

    return payload

def request_gpt4v(message, image, scene_id):
    payload = prepare_inputs(message, image, scene_id)
    while True:
        try:
            import pdb;pdb.set_trace()
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            res = response.json()['choices'][0]['message']['content']
            
            break
        except:
            continue
    return res

data = json.load(open("ScanQA_v1.0_val.json", 'r'))

count = 157
data = data[157:1500]
for d in data:
    message = "Question: " + d['question']
    scene_id = d['scene_id']
    data_path = f"scannet_gap50_val\\{scene_id}\\color-sampled"
    image_paths = glob.glob(f"{data_path}\\*")
    answer = request_gpt4v(message, image_paths[::6], scene_id)
    print(answer)
    with open(f'gpt\\gpt4v\\{count}.txt', 'w') as f:
        f.write(answer)
    count += 1