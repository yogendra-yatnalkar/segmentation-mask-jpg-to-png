import numpy as np
import cv2
import os
import time

def get_unique_rgb_colors(img):
    """
    returns a numpy array of unique rgb colors in input HxWxC image
    """
    reshaped_img = img.reshape((-1, img.shape[-1]))
    return np.unique(reshaped_img, axis = 0)

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
    return distance_from_label

def get_corrected_single_channel_mask(mask, all_labels_li):
    mask_shape = mask.shape
    multiple_distance_img_li = []
    print("Mask Shape: ", mask_shape)

    for label_li in all_labels_li:
        distance_img = get_mask_distance_from_rgb_label(mask, label_li)
        multiple_distance_img_li.append(distance_img)
        print(distance_img.shape)

    multiple_distance_arr = np.array(multiple_distance_img_li)
    print(multiple_distance_arr.shape)

    multiple_distance_arr = multiple_distance_arr.transpose(1,2,0)
    print(multiple_distance_arr.shape)

    label_arr = np.argmin(multiple_distance_arr, axis=-1)
    print('label_arr', label_arr, label_arr.shape)
    unique_colors, count = np.unique(label_arr, return_counts=True)
    print(unique_colors, count)

    new_mask = np.zeros(mask_shape)
    for iter_index, item in enumerate(all_labels_li):
        print("iter_index", iter_index)
        print("Item: ", item)
        found_index = np.where(label_arr == iter_index)
        new_mask[found_index] = item

    new_mask = new_mask.astype(np.uint8)
    return new_mask

if __name__ == "__main__":
    # getting the file-paths
    file_path = "D:\software-dev\jpg-to-png\data"
    jpg_file_name = "jpg-mask.jpg"
    png_file_name = "png-mask.png"
    jpg_img_path = os.path.join(file_path, jpg_file_name)
    png_img_path = os.path.join(file_path, png_file_name)

    # reading the mask
    jpg_mask = cv2.imread(jpg_img_path)
    png_mask = cv2.imread(png_img_path)

    # printing the unique colors from mask
    jpg_unique_colors = get_unique_rgb_colors(jpg_mask)
    png_unique_colors = get_unique_rgb_colors(png_mask)

    print("Number of unique colors in JPG Mask: ", len(jpg_unique_colors))
    print("JPG Unique Colors:\n ", jpg_unique_colors)
    print("\n", "="*30, "\n")
    print("Number of unique colors in PNG Mask: ", len(png_unique_colors))
    print("PNG Unique Colors:\n ", png_unique_colors)

    all_labels = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]]
    new_mask = get_corrected_single_channel_mask(jpg_mask, all_labels)

    img = png_mask
    reshaped_img = img.reshape((-1, img.shape[-1]))
    unique_colors, count = np.unique(reshaped_img, axis = 0, return_counts = True)
    print(unique_colors, count)

    print("COMPARING PNG AND NEW MASK")
    equal_check = np.array_equal(new_mask, png_mask)
    print("Equal Status Check: ", equal_check)


    # plotting the image
    cv2.imshow("jpg-mask", jpg_mask)
    cv2.imshow("png-mask", png_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()