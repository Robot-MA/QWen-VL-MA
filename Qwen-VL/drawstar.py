import cv2

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
    
    # Draw the center point
    point_color = (255,0,0)  # Red color
    point_radius = 5
    cv2.circle(image, center_point, point_radius, point_color, -1)  # -1 to fill the circle
    
    # Save the new image
    cv2.imwrite(save_path, image)

# Test the function
image_path = '/home/duanj1/CameraCalibration/LLMs/Qwen-VL/image/0.png'  # Replace with your image path
save_path = 'output_image.jpg'  # Path to save the new image
top_left = (355,333)  # Coordinates of the top-left corner of the bounding box
bottom_right = (498,475)  # Coordinates of the bottom-right corner of the bounding box

draw_bounding_box(image_path, top_left, bottom_right, save_path)
