from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import cv2
import os
import numpy as np
import math
from PIL import Image

def divide_image_into_patches(image_path, save_to_folder="./patches"):
    # Open the original image
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    width, height = img.size

    # Calculate dimensions of each patch
    patch_width = width // 2
    patch_height = height // 2

    # Create patches and save them
    img1 = img.crop((0, 0, patch_width, patch_height))
    img1.save(f"{save_to_folder}/patch1.jpg")

    img2 = img.crop((patch_width, 0, width, patch_height))
    img2.save(f"{save_to_folder}/patch2.jpg")

    img3 = img.crop((0, patch_height, patch_width, height))
    img3.save(f"{save_to_folder}/patch3.jpg")

    img4 = img.crop((patch_width, patch_height, width, height))
    img4.save(f"{save_to_folder}/patch4.jpg")

if __name__ == "__main__":
    image_path = "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/image/kitchen3.png"  # Replace with the path to your image
    divide_image_into_patches(image_path)

torch.manual_seed(1234)
from translate import Translator
def combine_images(image_paths):
    # Load the images
    images = [cv2.imread(path) for path in image_paths]
    
    # Check if any of the images couldn't be loaded
    for i, image in enumerate(images):
        if image is None:
            print(f"Error: Couldn't read image from {image_paths[i]}")
            return None

    # Ensure all images are of the same size
    # Assuming first image dimensions represent the standard dimensions
    std_height, std_width = images[0].shape[:2]

    for i in range(1, len(images)):
        height, width = images[i].shape[:2]
        if height != std_height or width != std_width:
            print("Resizing images to match the dimensions of the first image.")
            images[i] = cv2.resize(images[i], (std_width, std_height))

    # Combine top images (1 and 2)
    top_row = np.hstack((images[0], images[1]))

    # Combine bottom images (3 and 4)
    bottom_row = np.hstack((images[2], images[3]))

    # Combine top and bottom
    combined_image = np.vstack((top_row, bottom_row))

    return combined_image
def translate_english_to_chinese(english_text):
    translator= Translator(to_lang="zh")
    translation = translator.translate(english_text)
    return translation

def translate_chinese_to_english(chinese_text):
    translator = Translator(from_lang="zh", to_lang="en")
    translation = translator.translate(chinese_text)
    return translation

def draw_and_save_bounding_boxes(box1, box2, image_path, output_path):
    """
    Draw bounding boxes on an image and save it.
    
    Parameters:
    - box1: tuple (x1, y1, x2, y2) for the first bounding box
    - box2: tuple (x1, y1, x2, y2) for the second bounding box
    - image_path: str, the path to the input image
    - output_path: str, the path to save the output image
    
    Returns: None
    """
    
    # Read the image from file
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    # Draw the first bounding box
    cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)
    
    # Draw the second bounding box
    cv2.rectangle(image, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 2)
    
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)


# Example usage
image_path = "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/image/kitchen3.png"  # Replace with your input image path
output_folder = "patches"  # Replace with your output folder name

# create_patches(image_path, output_folder)

# prompt1 = "How many drawers on the cabinet that is pulled opened in the image?"
# chinese_prompt1 = "图像里面有什么东西？"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device框出图中
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
for i in range(1,5):
    imagepath= '/home/duanj1/CameraCalibration/LLMs/Qwen-VL/patches/patch'+str(i)+'.jpg'
    query = tokenizer.from_list_format([
        {'image': imagepath}, # Either a local path or an url
        {'text': "图像里面有什么东西？"},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    prompt2 = "框出图中"+str(response)+"具柜的位置"
    response, history = model.chat(tokenizer, query=prompt2, history=history)
    print(response)
    # <ref>击掌</ref><box>(536,509),(588,602)</box>
    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save(str(i)+'.jpg')
    else:
        print("no box")

image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]  # Replace with your image paths
combined_image = combine_images(image_paths)

if combined_image is not None:
    cv2.imwrite("combined_image.jpg", combined_image)
    print("Successfully combined images and saved as 'combined_image.jpg'")
# box1 = (10,466 ,944,945)  # x1, y1, x2, y2
# # box1 = (446,10., 945,944)  # x1, y1, x2, y2

# box2 = (194,116, 517,430)  # x1, y1, x2, y2
# image_path = "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/image/kitchen1.jpg"  # Replace with the path to your image
# output_path = "./2.jpg"  # Replace with the path where you want to save the output image

# draw_and_save_bounding_boxes(box1, box2, image_path, output_path)