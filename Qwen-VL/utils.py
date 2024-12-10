from transformers import pipeline
import base64
import cv2
import os


def english_to_chinese(text):
    translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
    translation = translator(text, max_length=512)
    return translation[0]['translation_text']

# Google translator which has a limit
# def english_to_chinese(text):
#     translator = Translator(to_lang="zh")
#     translation = translator.translate(text)
#     return translation

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def draw_bounding_box(image_path, top_left, bottom_right, save_path):
    # Read the image from the file
    image = cv2.imread(image_path)
    
    # Draw the bounding box
    color = (0, 255, 0)  # Green color
    thickness = 2  # Line thickness
    # cv2.rectangle(image, top_left, bottom_right, color, thickness)
    
    # Calculate the center point of the bounding box
    center_x = int((top_left[0] + bottom_right[0]) / 2)
    center_y = int((top_left[1] + bottom_right[1]) / 2)
    center_point = (center_x, center_y)
    # print(center_point)
    
    # Draw the center point
    point_color = (255, 255, 0)  # Yellow color
    point_radius = 5
    cv2.circle(image, center_point, point_radius, point_color, -1)  # -1 to fill the circle
    
    # Save the new image
    cv2.imwrite(save_path, image)

def get_most_recent_image(folder_path):
    # List all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    # Filter out files that are not images (assuming image files have extensions like .jpg, .png, etc.)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    # Sort the images by creation time and get the most recent one
    if not image_files:
        return None  # Return None if no images are found
    most_recent_image = max(image_files, key=os.path.getctime)
    return most_recent_image

def get_completion(prompt, model="gpt-4-1106-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,
    )

    return response.choices[0].message["content"]