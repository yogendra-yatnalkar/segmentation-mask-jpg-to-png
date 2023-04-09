import numpy as np
import cv2
import os
import time


def get_color_distance(color_image_1, color_image_2):
    color_image_1 = color_image_1.astype(np.float64)
    color_image_2 = color_image_2.astype(np.float64)
    red_average = (color_image_1[:, :, 2] + color_image_2[:, :, 2]) / 2
    b_difference = color_image_1[:, :, 0] - color_image_2[:, :, 0]
    g_difference = color_image_1[:, :, 1] - color_image_2[:, :, 1]
    r_difference = color_image_1[:, :, 2] - color_image_2[:, :, 2]

    red_component_1 = 2 + (red_average / 256)
    red_component_2 = 2 + ((255 - red_average) / 256)

    squared_color_distance = (
        (red_component_1 * (r_difference**2))
        + (4 * (g_difference**2))
        + (red_component_2 * (b_difference**2))
    )

    color_distance = np.sqrt(squared_color_distance)
    return color_distance

def get_mask_distance_from_rgb_label(mask, label_li):
    """
    Inputs:
        - mask: HxWxC jpg currpted mask
        - label_value: List containing RGB value of the actual/true 
    Returns:
        - Distance of every pixel of mask to the given label value
    """
    label_arr = np.array(label_li)
    label_img = np.zeros(mask.shape) + label_arr
    label_img = label_img.astype(mask.dtype)
    distance_from_label = get_color_distance(mask, label_img)
    pass

def get_rgb_mask_from_label(label_arr, mask_shape=(1966, 1966, 3)):
    mask = np.zeros(mask_shape)

    index = np.where(label_arr == 0)
    mask[index] = [255, 0, 0]
    del index

    index = np.where(label_arr == 1)
    mask[index] = [0, 255, 0]
    del index

    index = np.where(label_arr == 2)
    mask[index] = [0, 0, 255]
    del index

    index = np.where(label_arr == 3)
    mask[index] = [255, 255, 255]
    del index

    index = np.where(label_arr == 4)
    mask[index] = [255, 255, 255]
    del index

    mask = mask.astype(np.uint8)
    return mask


def sorted_unique_colors_on_count(mask):
    vertical_mask = np.vstack(mask)
    unique_colors, count = np.unique(vertical_mask, axis=0, return_counts=True)
    sort_index = np.argsort(count)

    sorted_unique_colors = unique_colors[sort_index]
    sorted_count = count[sort_index]

    return sorted_unique_colors, sorted_count, unique_colors, count


if __name__ == "__main__":
    mask_dir_path = "./masks/"
    save_path = "./updated_masks/"
    mask_names = os.listdir(mask_dir_path)

    for mask_name in mask_names:
        start = time.time()

        new_mask_name = mask_name.rsplit(".", 1)[0] + ".png"
        mask_path = os.path.join(mask_dir_path, mask_name)
        print(mask_path)
        mask = cv2.imread(mask_path)

        blue_img = np.zeros(mask.shape) + [255, 0, 0]
        blue_img = blue_img.astype(mask.dtype)

        green_img = np.zeros(mask.shape) + [0, 255, 0]
        green_img = green_img.astype(mask.dtype)

        red_img = np.zeros(mask.shape) + [0, 0, 255]
        red_img = red_img.astype(mask.dtype)

        white_img = np.zeros(mask.shape) + [255, 255, 255]
        white_img = white_img.astype(mask.dtype)

        black_img = np.zeros(mask.shape) + [0, 0, 0]
        black_img = black_img.astype(mask.dtype)

        distance_from_blue = get_color_distance(mask, blue_img)
        distance_from_green = get_color_distance(mask, green_img)
        distance_from_red = get_color_distance(mask, red_img)
        distance_from_white = get_color_distance(mask, white_img)
        distance_from_black = get_color_distance(mask, black_img)

        distance_array = np.dstack(
            (
                distance_from_blue,
                distance_from_green,
                distance_from_red,
                distance_from_white,
                distance_from_black,
            )
        )
        label_array = np.argmin(distance_array, axis=-1)

        unique_colors, count = np.unique(label_array, return_counts=True)
        print(unique_colors, count)

        new_mask = get_rgb_mask_from_label(label_array)
        cv2.imwrite(os.path.join(save_path, new_mask_name), new_mask)

        (
            sorted_mask_unq_colors,
            sorted_mask_col_count,
            mask_unq_colors,
            mask_col_count,
        ) = sorted_unique_colors_on_count(mask)
        print(mask_unq_colors, mask_col_count)
        print(sorted_mask_unq_colors, sorted_mask_col_count)

        print("\n--------------------------------------\n")

        (
            sorted_new_mask_unq_colors,
            sorted_new_mask_col_count,
            new_mask_unq_colors,
            new_mask_col_count,
        ) = sorted_unique_colors_on_count(new_mask)
        print(new_mask_unq_colors, new_mask_col_count)
        print(sorted_new_mask_unq_colors, sorted_new_mask_col_count)

        end = time.time()
        print("\nTIME REQUIRED: ", end - start, " sec\n")