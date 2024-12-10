import os
import glob

def find_latest_images(folder_path):
    # Search for image files (jpg, png, etc.) in the folder
    image_files = glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]')) + \
                  glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]'))

    # Sort the files by modification time in descending order
    image_files.sort(key=os.path.getmtime, reverse=True)

    # Check the number of images and return accordingly
    if len(image_files) >= 2:
        return image_files[:2]
    elif len(image_files) == 1:
        return [image_files[0], image_files[0]]  # Return the same image path twice
    else:
        return [None, None]  # No images found

# Example usage
folder_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/save_frames'
latest_images = find_latest_images(folder_path)
print("Most Recent Image:", latest_images[0] or "No image found.")
print("Second Most Recent Image:", latest_images[1] or "No image found.")