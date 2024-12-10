import cv2
import os
import glob
import base64
import requests

def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path) and file_path.endswith(('.jpg', '.jpeg', '.png')):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error occurred while deleting file {file_path}: {e}")


def extract_frames(video_path, output_folder):
    clear_folder(output_folder)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_capture = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
    captured_frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frames_to_capture:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            captured_frames.append(frame_path)

    cap.release()
    return captured_frames

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def GPTV4_Verification(prompt_text, video_path, output_folder, api_key):
    # Extract frames from the video
    frame_paths = extract_frames(video_path, output_folder)

    # Encoding all images
    base64_images = [encode_image(path) for path in frame_paths]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Preparing the messages part of the payload
    image_messages = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                      for image in base64_images]

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}] + image_messages
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    output = response.json()['choices'][0]['message']['content']
    return str(output)

# Example usage
video_path = '/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/close_box_w80000_s2_succ.mp4'
output_folder = '/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/frames'
api_key = os.getenv("OPENAI_API_KEY")

# Read prompt text from a file
with open('/home/duanj1/CameraCalibration/LLMs/LLM_Progprompt/reflection.txt', 'r') as file:
    prompt_text = file.read()

result = GPTV4_Verification(prompt_text, video_path, output_folder, api_key)
print(result)
