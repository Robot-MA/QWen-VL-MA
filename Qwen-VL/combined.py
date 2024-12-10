from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import cv2
import os
import numpy as np
import math
from PIL import Image

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

image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]  # Replace with your image paths
combined_image = combine_images(image_paths)

if combined_image is not None:
    cv2.imwrite("combined_image.jpg", combined_image)
    print("Successfully combined images and saved as 'combined_image.jpg'")