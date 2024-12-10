import cv2
import numpy as np
import os
import glob

rlbench_env = "/home/duanj1/m2t2/manipulate_anything/RLBench"
qwen_env = "/home/duanj1/CameraCalibration/"


def find_latest_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]')) + \
                  glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]'))
    image_files.sort(key=os.path.getmtime, reverse=True)
    if len(image_files) >= 2:
        return image_files[:2]
    elif len(image_files) == 1:
        return [image_files[0], image_files[0]]
    else:
        return [None, None]

def label_and_combine_images(image_path1, image_path2):
    # Load the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Check if images were loaded successfully
    if image1 is None or image2 is None:
        print("Error loading images.")
        return
    
    # Define circle center and radius
    center = (50, 50)
    radius = 30
    
    # Draw white circles on both images
    cv2.circle(image1, center, radius, (255, 255, 255), -1) # -1 fills the circle
    cv2.circle(image2, center, radius, (255, 255, 255), -1)
    
    # Label each image with numbers inside the circle. Adjusted positions to center the text.
    cv2.putText(image1, "1", (center[0] - 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image2, "2", (center[0] - 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Combine images horizontally
    combined_image = np.hstack((image1, image2))
    output_path = qwen_env+'LLMs/LLM_Progprompt/combined_image_front.jpg' # Specify the output file path

    # Display the combined image
    cv2.imwrite(output_path, combined_image)


# Example usage
# print("Combining....")
folder_path = rlbench_env+'/save_frames'
latest_images = find_latest_images(folder_path)
if latest_images[0] is not None and latest_images[1] is not None:
    label_and_combine_images(latest_images[1], latest_images[0])
else:
    print("Not enough images found.")
