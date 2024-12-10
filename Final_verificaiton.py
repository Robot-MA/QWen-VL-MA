import os
import base64
import shutil
import cv2
import natsort

def extract_frames_from_video(video_path, frame_folder, every_n_frames=1):
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % every_n_frames == 0:
            frame_path = os.path.join(frame_folder, f"frame{count:05d}.jpg")
            cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1

def get_sampled_image_paths(folder_path, num_samples=10):
    all_image_paths = [os.path.join(folder_path, f) for f in natsort.natsorted(os.listdir(folder_path)) if f.endswith(('.png', '.jpg', '.jpeg'))]
    N = len(all_image_paths)
    step = max(1, N // num_samples)
    sampled_image_paths = all_image_paths[::step][:num_samples]
    return sampled_image_paths

def save_sampled_images(sampled_image_paths, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for image_path in sampled_image_paths:
        shutil.copy(image_path, destination_folder)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

# Specify the video file path and folders
video_folder = '/home/duanj1/peract_sim_env/peract/logs/pick_and_lift/PERACT_BC/seed4/videos'  # Replace with the path to your folder containing videos
frame_folder = '/home/duanj1/CameraCalibration/LLMs/video_folder'  # Replace with your frame folder path
destination_folder = '/home/duanj1/CameraCalibration/LLMs/frames'  # Destination folder path
counter=0

for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):  # Check for video files, e.g., .mp4
        video_path = os.path.join(video_folder, video_file)
        clear_folder(destination_folder)

        # Extract frames from the video
        extract_frames_from_video(video_path, frame_folder)

        # Get sampled image paths from the frame folder
        sampled_image_paths = get_sampled_image_paths(frame_folder)

        # Save sampled images to the destination folder
        save_sampled_images(sampled_image_paths, destination_folder)

        # Clear the frame folder
        clear_folder(frame_folder)

        import os
        import base64
        import requests

        # OpenAI API Key
        api_key = "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"

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Directory containing your images
        image_directory = destination_folder 

        # Get a list of image paths in the directory, limit to first 10 images
        image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.jpg')][:10]
        print(image_paths)
        # Encoding all images
        base64_images = [encode_image(path) for path in image_paths]

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
                    "content": [{"type": "text", "text": "Based on the sequence of frames, did the robot successfully pick up and lift the red cube?"}] + image_messages
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output= response.json()['choices'][0]['message']['content']
        counter = counter +1
        print(str(counter)+':'+str(output))
