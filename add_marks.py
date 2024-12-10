from PIL import Image, ImageDraw, ImageFont
import os, argparse
import numpy as np

def add_number_mark_size(img: Image, 
                         mark_num: int, 
                         mark_center: tuple,
                         mark_size = 80) -> Image:
    assert mark_size in [60], "mark_size should be in [60]"
    assert 0 <= mark_num and mark_num <= 999, "mark_num should be in the range [0, 999]"

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw the red circle background
    radius = mark_size // 2    
    draw.ellipse([(mark_center[0] - radius, mark_center[1] - radius),
                  (mark_center[0] + radius, mark_center[1] + radius)],
                 fill='red', outline='red')

    # Draw the number mark
    if mark_size == 60:
        if mark_num <= 9:
            horizontal_offset = 18
            vertical_offset = 5
            font_size = int(radius*1.3)
        elif mark_num <= 99:
            horizontal_offset = 5
            vertical_offset = 5
            font_size = int(radius*1.3)
        else:
            horizontal_offset = 5
            vertical_offset = 10
            font_size = int(radius)
    
    text_position = (mark_center[0] - radius + horizontal_offset, mark_center[1] - radius + vertical_offset)
    text_color = 'white'
    font = ImageFont.truetype("font/Rubik-Bold.ttf", size=font_size)
    draw.text(text_position, str(mark_num), fill=text_color, font=font)
    
    return img


def find_center(mask: np.array) -> tuple:
    # find center1
    left_bound = mask.shape[1]
    right_bound = 0
    for x in range(mask.shape[1]):
        if mask[:, x].any():
            left_bound = min(left_bound, x)
            right_bound = max(right_bound, x)
    x_center = (left_bound + right_bound) // 2
    
    y_upper_bound = 0
    y_lower_bound = mask.shape[0]
    exist_y = []
    # go through the x_center column, find all the points that exist in the mask, get its median position (not average, because median makes sure existence)
    for y in range(mask.shape[0]):
        if mask[y, x_center]:
            y_upper_bound = max(y_upper_bound, y)
            y_lower_bound = min(y_lower_bound, y)
            exist_y.append(y)
    if exist_y == []:
        center1 = None
    else:
        y_center = np.median(exist_y).astype(int)
        center1 = (x_center, y_center)

    # find center2
    upper_bound = 0
    lower_bound = mask.shape[0]
    for y in range(mask.shape[0]):
        if mask[y, :].any():
            upper_bound = max(upper_bound, y)
            lower_bound = min(lower_bound, y)
    y_center = (upper_bound + lower_bound) // 2

    x_left_bound = mask.shape[1]
    x_right_bound = 0
    exist_x = []
    # go through the y_center row, find all the points that exist in the mask, get its median position (not average, because median makes sure existence)
    for x in range(mask.shape[1]):
        if mask[y_center, x]:
            x_left_bound = min(x_left_bound, x)
            x_right_bound = max(x_right_bound, x)
            exist_x.append(x)
    if exist_x == []:
        center2 = None
    else:
        x_center = np.median(exist_x).astype(int)
        center2 = (x_center, y_center)

    if center1 is not None and center2 is not None:
        center_avg = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
    else:
        center_avg = None

    # if everything goes wrong, simply use naive center
    center_naive = ((left_bound + right_bound) // 2, (upper_bound + lower_bound) // 2)

    if (center_avg is not None) and mask[center_avg[1], center_avg[0]]:
        center = center_avg
    elif (center1 is not None) and mask[center1[1], center1[0]]:
        center = center1
    elif (center2 is not None) and mask[center2[1], center2[0]]:
        center = center2
    else:
        center = center_naive

    return center



def add_seg_mask(img: Image, vis_img: Image, mask: np.array, mask_rate: float=0.6) -> Image:

    assert mask_rate >= 0 and mask_rate <= 1, "mask_rate should be in the range [0, 1]"

    # use mask to process vis_img, only keep those pixel=True
    vis_img_masked = np.array(vis_img)
    vis_img_masked[mask == False] = 0
    img_masked = np.array(img)
    img_masked[mask == False] = 0
    img_unmasked = np.array(img)
    img_unmasked[mask == True] = 0

    img = mask_rate * vis_img_masked + (1 - mask_rate) * img_masked + img_unmasked
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)



def main(args):

    task_id = args.task_id
    total_id = 3

    root_dir = ''
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    # list sub_dir in a sorted way
    sub_dir_list = os.listdir(root_dir)
    sub_dir_list.sort()

    for scene_id, sub_dir in enumerate(sub_dir_list):
        if scene_id % total_id != task_id:
            continue
        print(f'Processing {sub_dir}...')

        current_instance_dir = os.path.join(root_dir, sub_dir, 'instance-sampled')
        current_color_dir = os.path.join(root_dir, sub_dir, 'color-sampled')
        current_marked_dir = os.path.join(root_dir, sub_dir)
        if not os.path.exists(current_marked_dir):
            os.mkdir(current_marked_dir)
        
        current_marked_dir = os.path.join(current_marked_dir, 'marked-sampled')
        if not os.path.exists(current_marked_dir):
            os.mkdir(current_marked_dir)


        # load all instance img path (total 8), sort, than load as Image
        img_id_list = os.listdir(current_instance_dir)
        img_id_list = [int(img_path.split('.')[0]) for img_path in img_id_list]
        img_id_list.sort()


        instance_img_list = [os.path.join(current_instance_dir, f'{img_id}.png') for img_id in img_id_list]
        instance_img_list = [Image.open(img_path) for img_path in instance_img_list]

        color_img_list = [os.path.join(current_color_dir, f'{img_id}.jpg') for img_id in img_id_list]
        color_img_list = [Image.open(img_path) for img_path in color_img_list]

        instance_arr_flatten = [np.array(instance_img).flatten() for instance_img in instance_img_list]
        instance_unique_counts_list = [np.unique(instance_arr, return_counts=True) for instance_arr in instance_arr_flatten]
        instance_unique_counts_list = [(unique_arr[1:], counts_arr[1:]) for unique_arr, counts_arr in instance_unique_counts_list]
        labels_per_img = 8  # each img will have 8 labels marked

        for idx, img_id in enumerate(img_id_list):
            instance_unique_counts = instance_unique_counts_list[idx]

            marked_img = color_img_list[idx].copy()
            
            # sort label according to the second element of the tuple in instance_unique_counts
            sorted_label = sorted(zip(instance_unique_counts[0], instance_unique_counts[1]), key=lambda x: x[1], reverse=True)
            num_label_to_add = min(labels_per_img, len(sorted_label))
            for i in range(num_label_to_add):
                label = sorted_label[i][0]
                mask = (np.array(instance_img_list[idx]) == label)
                center = find_center(mask)
                marked_img = add_number_mark_size(marked_img, label-1, center, mark_size=60)

            marked_img.save(os.path.join(current_marked_dir, f'{img_id}_marked.jpg'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add marks to the color images')
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()
    main(args)
