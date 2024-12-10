from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
from translate import Translator
import ast
import os
import glob
from transformers import pipeline
import subprocess
import os
import openai
import requests
import torch
import random
from utils import get_completion,english_to_chinese, encode_image,draw_bounding_box,get_most_recent_image


# Generate a random 4-digit seed
random_seed = random.randint(1000, 9999)
# print(f"Random Seed: {random_seed}")
api_key = os.getenv("OPENAI_API_KEY")
rlbench_env = "/home/duanj1/m2t2/manipulate_anything/RLBench"
qwen_env = "/home/duanj1/CameraCalibration/"

imagepath=str(rlbench_env)+'/saved_image.png'

# Set the random seed for PyTorch
torch.manual_seed(random_seed)

# Path to your image
image_path = imagepath
print(image_path)
base64_image = encode_image(image_path)
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key }"
}

def GPTV4_Verification_Sinlge(prompt_text):
    def find_latest_image(folder_path):
        # Search for image files (jpg, png, etc.) in the folder
        image_files = glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]')) + \
                    glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                    glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]'))

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Return the most recent image if available
        return image_files[0] if image_files else None

    # Example usage
    subprocess.run(["python", qwen_env+"LLMs/LLM_Progprompt/combine_image_front.py"])
    subprocess.run(["python", qwen_env+"LLMs/LLM_Progprompt/combine_image_wrist.py"])
    subprocess.run(["python", qwen_env+"LLMs/LLM_Progprompt/combine_image_left.py"])
    subprocess.run(["python", qwen_env+"LLMs/LLM_Progprompt/combine_image_right.py"])
    # folder_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/combined_image.png'
    print((GPT_4V_Viewpoint()))
    verification_viewpoint = int(GPT_4V_Viewpoint().strip('[]'))
    option=['front', 'wrist', 'left', 'right']
    folder_path = qwen_env+'LLMs/LLM_Progprompt/combined_image_'+str(option[verification_viewpoint])+'.jpg'

    most_recent_image = folder_path
    print("Most Recent Image:", most_recent_image or "No image found.")

    if most_recent_image:
        # Encoding the image
        encoded_image = encode_image(most_recent_image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Preparing the message part of the payload with the single image
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}, image_message]
                }
            ],
            "max_tokens": 300
        }

        # Sending the request (assuming requests is imported and api_key is defined)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        output = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print('Sub-task Verification Prediction:' + str(output))
        # with open('/home/duanj1/m2t2/manipulate_anything/RLBench/log.txt', 'a') as file:
        with open(str(rlbench_env)+'/log.txt', 'a') as file:

            file.write("Prediction output: "+str(output)  + "\n")
        return str(output)
    
    else:
        return "No image found to process."

folder_path_wrist = str(rlbench_env)+'/save_frames_wrist'
most_recent_image = get_most_recent_image(folder_path_wrist)
if most_recent_image:
    imagepath_wrist=str(most_recent_image)
    print(imagepath_wrist)
else:
    print("No images found in the folder.")

folder_path_left=str(rlbench_env)+'/save_frames_left'
most_recent_image_left = get_most_recent_image(folder_path_left)
if most_recent_image_left:
    imagepath_left=str(most_recent_image_left)
    print(most_recent_image_left)
else:
    print("No images found in the folder.")

folder_path_right=str(rlbench_env)+'/save_frames_right'
most_recent_image_right = get_most_recent_image(folder_path_right)
if most_recent_image_right:
    imagepath_right=str(most_recent_image_right)
    print(most_recent_image_right)
else:
    print("No images found in the folder.")


def GPT_4V_Viewpoint():
    file_path = str(rlbench_env)+'/action_info.txt'
    with open(file_path, 'r') as file:
        content = file.read()
        data_dict = ast.literal_eval(content)
    next_verification=data_dict['verification'][0]
    next_action = data_dict['primitive_actions'][0]
    next_obj =  data_dict['Objects'][0]
    # next_SoM = data_dict['SoM']

    # folder_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/combined_image.png'

    most_recent_image = str(rlbench_env)+'/combined_image_all.png'
    
    prompt = f"""
    "type": "text",
    "text": "There is a picture containing 4 frames of a robot manipulation scene at the same time step but observed from four different camera viewpoints.",
    "text": "Each frame is annotated with a number on the top left corner of the image, numbered from 0 to 3.",
    "text": "The frame with number 0 annotated refers to the front viewpoint, number 1 refers to the wrist viewpoint, number 2 refers to the left shoulder viewpoint, and number 3 refers to the right shoulder viewpoint.",
    "text": "Given that the robot agent is currently performing the sub-task: {next_action} {next_obj}, compare the four frames and select the one viewpoint that offers the least obstructed view  and could see the robot arm performing {next_action} {next_obj}. This will help in evaluating the success of the verification condition: {next_verification}.",
    "text": "Output should only be one number between 0 and 3 representing the different viewpoints, and written in a list format, for example .",
    "text": "For example, if the front viewpoint (0) provides the clearest view, the output should be: [0]."
    """
    # prompt_input = prompt+ str(data_dict['verification'][0])
    prompt_input = prompt
    print("MT-Viewpoint Selection...")
    # print("MT-VIewpoint Selection Prompt (Verification):")
    # print(prompt_input)


    if most_recent_image:
        # Encoding the image
        encoded_image = encode_image(most_recent_image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Preparing the message part of the payload with the single image
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_input}, image_message]
                }
            ],
            "max_tokens": 300
        }

        # Sending the request (assuming requests is imported and api_key is defined)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print("Ouptut: "+str(output))

        return str(output)
    else:
        return "No image found to process."






tokenizer = AutoTokenizer.from_pretrained(qwen_env+"LLMs/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(qwen_env+"LLMs/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(qwen_env+"LLMs/Qwen-VL-Chat", trust_remote_code=True)
file_path = str(rlbench_env)+'/action_info.txt'
with open(file_path, 'r') as file:
    content = file.read()
    data_dict = ast.literal_eval(content)
next_verification=data_dict['verification'][0]
next_action = data_dict['primitive_actions'][0]
next_obj =  data_dict['Objects'][0]
# next_SoM = data_dict['SoM']




# Now, save the modified dictionary back to the original file.
with open(file_path, 'w') as file:
    file.write(str(data_dict))

file_path = str(rlbench_env)+'/action_info.txt'
with open(file_path, 'r') as file:
    content = file.read()
    data_dict = ast.literal_eval(content)
next_verification=data_dict['verification'][0]

content = f"""
"type": "text",
"text": f"There is an image containing 2 frames of robot manipulation scene, one frame being the past time-step and one being the current time-step observation",
"text": f"Each frame are annotated with a number on the top left corner of the image.",
"text": f"The frame with number 1 annotated on the top left refers to frame from the previous time-step, number 2 refers to the current time-step.",
"text": f"If the two image are identical, then just use the number 1 frame to determine the success of the verification condition.",
"text": f"Evaluate the 2 frames concatened into the image, and determine if the verification condition of {next_verification} has succeed?",
"text": f"Output should only be either Yes or No."
"""

verification_prompt=str(content)
with open(str(rlbench_env)+'/log.txt', 'a') as file:
    file.write("Verification: "+str(next_verification)  + "\n")
print("NEXT_VERIFICATION: "+str(next_verification))

response3=GPTV4_Verification_Sinlge(verification_prompt)


query = tokenizer.from_list_format([
        {'image': imagepath}, # Either a local path or an url
        {'text': verification_prompt},
    ])


response, history = model.chat(tokenizer, query, history=None)
translator = Translator(to_lang="en", from_lang="zh")

# Translate the input text to English
# translated_text = translator.translate(str(response))
translated_text = (str(response))


if response3 == 'Yes':
    # Remove the specified primitive_action and objects from the dictionary
    if str(next_action) in data_dict['primitive_actions']:
        data_dict['primitive_actions'].remove(str(next_action))
    data_dict['Objects'].remove(str(next_obj))
    data_dict['verification'].remove(str(next_verification))
    data_dict['predict'] = int(data_dict['predict'])+1

    # Now, save the modified dictionary back to the original file.
    with open(file_path, 'w') as file:
        file.write(str(data_dict))
else:
    print("Translation is not 'Yes', so action_info won't be updated.")

obj=english_to_chinese(data_dict['Objects'][0])
prompt1 = '框出图中只有一个'+obj+'的位置'
next_act=data_dict['primitive_actions'][0]
data_dict['pick'] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
with open(str(rlbench_env)+'/action_info.txt', 'w') as file:
    file.write(str(data_dict))


if next_act == 'pick':

    query = tokenizer.from_list_format([
            {'image': imagepath}, # Either a local path or an url
            {'text': prompt1},
        ])

    response, history = model.chat(tokenizer, query=query, history=None)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('save1.jpg')
    else:
        print("no box")

    query = tokenizer.from_list_format([
            {'image': imagepath_wrist}, # Either a local path or an url
            {'text': prompt1},
        ])

    response, history = model.chat(tokenizer, query=query, history=None)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('save2.jpg')
    else:
        print("no box")

    query = tokenizer.from_list_format([
            {'image': imagepath_left}, # Either a local path or an url
            {'text': prompt1},
        ])

    response, history = model.chat(tokenizer, query=query, history=None)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('save3.jpg')
    else:
        print("no box")
    query = tokenizer.from_list_format([
            {'image': imagepath_right}, # Either a local path or an url
            {'text': prompt1},
        ])

    response, history = model.chat(tokenizer, query=query, history=None)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('save4.jpg')
    else:
        print("no box")

    


else:
    query = tokenizer.from_list_format([
            {'image': imagepath}, # Either a local path or an url
            {'text': prompt1},
        ])

    response, history = model.chat(tokenizer, query=query, history=None)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('save1.jpg')
    else:
        print("no box")

 

# Test the function
with open('single_coordinate.txt', 'r') as f:
    line = f.readline().strip()
    x1, y1, x2, y2 = map(int, line.split(','))
save_path = 'output_image.jpg'  # Path to save the new image
top_left = (x1, y1)  # Coordinates of the top-left corner of the bounding box
bottom_right = ( x2, y2)  # Coordinates of the bottom-right corner of the bounding box

draw_bounding_box(imagepath, top_left, bottom_right, save_path)

