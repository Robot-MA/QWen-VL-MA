import cv2

def draw_box(image_path, top_left, bottom_right, save_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Draw a rectangle on the image
    color = (0, 0, 255)  # Color in BGR (Red)
    thickness = 2
    cv2.rectangle(img, top_left, bottom_right, color, thickness)

    # Save the image
    cv2.imwrite(save_path, img)

# Usage
image_path = '/home/duanj1/CameraCalibration/LLMs/Qwen-VL/patches/patch4.jpg'
save_path = './image.jpg'  # Specify the path where you want to save the image
top_left = (0,81)  # (x, y)
bottom_right = (384,550)  # (x, y)
draw_box(image_path, top_left, bottom_right, save_path)
