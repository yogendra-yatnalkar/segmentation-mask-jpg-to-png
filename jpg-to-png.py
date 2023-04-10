import numpy as np
import cv2
import os

def get_color_distance(color_image_1, color_image_2):
    """
    Input:
        - color_image_1 (dtype: np.uint8): RGB/BGR image input
        - color_image_2 (dtype: np.uint8): RGB/BGR image input
    Output: 
        - color_distance (dtype: float): mxn numpy nd array where each value 
                    represents distance between 2 pixels from the 2 input 
                    rgb images
    
        The function has been written after reading and analyzing this article:
        https://www.compuphase.com/cmetric.htm
    """
    # convert the input images to float
    color_image_1 = color_image_1.astype(np.float64)
    color_image_2 = color_image_2.astype(np.float64)

    # average of red color from both the images
    red_average = (color_image_1[:, :, 2] + color_image_2[:, :, 2]) / 2

    # channel wise difference in color values
    b_difference = color_image_1[:, :, 0] - color_image_2[:, :, 0]
    g_difference = color_image_1[:, :, 1] - color_image_2[:, :, 1]
    r_difference = color_image_1[:, :, 2] - color_image_2[:, :, 2]

    # computing the final formulae
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
    Input:
        - mask: HxWxC jpg currpted mask
        - label_li: Single original RGB label value example: [255, 0, 0]
    Output:
        - distance_from_label (dtype: np.float64): HxW np nd array
    Distance of every pixel of mask to the given label value
    """
    label_arr = np.array(label_li)

    # create a new rgb image of the size of input mask image.
    label_img = np.zeros(mask.shape) + label_arr
    label_img = label_img.astype(mask.dtype)

    # get the distance between jpg currpted image and the created label image
    distance_from_label = get_color_distance(mask, label_img)
    return distance_from_label

def get_corrected_single_channel_mask(mask, all_labels_li):
    """
    Input:
        - mask (dtype: np.uint8): HxWxC jpg currpted mask
        - all_labels_li (dytpe: python list): List containing all the unique 
                    non-corrupted RGB label values.
                    Example: [[255, 0, 0], [255, 255, 255], [0, 0, 0]]
    Output: 
        - new_mask (dtype: np.uint8): HxWxC jpg distortion corrected mask
    The function takes the jpg image as input and returns a corrected
    image as output. The corrected image will not have any distortion created
    due to jpg compression.

    Note: Dont save the corrected image as JPG/JPEG image again :)
    """
    # constants
    mask_shape = mask.shape
    multiple_distance_img_li = []

    # For all the labels, create a label image as shown in "get_mask_distance_from_rgb_label"
    # function. Later, get the distance matrix between jpg image and each label. 
    # Store all such matrices in a python list and later create a numpy array 
    # out of that list 
    for label_li in all_labels_li:
        distance_img = get_mask_distance_from_rgb_label(mask, label_li)
        multiple_distance_img_li.append(distance_img)
    multiple_distance_arr = np.array(multiple_distance_img_li)

    # if the mask shape is HxWxC and there are n labels,
    # then the shape of multiple_distance_arr after transpose will be: 
    # HxWxn. Each n will be the distance matrix from a corresponding label image
    multiple_distance_arr = multiple_distance_arr.transpose(1,2,0)

    # Find the argmin from the distance matrix
    # label_arr will be a HxW 1D matrix.
    label_arr = np.argmin(multiple_distance_arr, axis=-1)

    # For the 1D lablel_arr matrix, assign its corresponding RGB label 
    # and create a RGB jpg corrected image. 
    new_mask = np.zeros(mask_shape)
    for iter_index, item in enumerate(all_labels_li):
        found_index = np.where(label_arr == iter_index)
        new_mask[found_index] = item
    
    # making the corrected image of dtype np.uint8 
    new_mask = new_mask.astype(np.uint8)
    return new_mask

if __name__ == "__main__":
    # getting the file-paths
    file_path = "./data"
    jpg_file_name = "jpg-mask.jpg"
    png_file_name = "png-mask.png"
    jpg_img_path = os.path.join(file_path, jpg_file_name)
    png_img_path = os.path.join(file_path, png_file_name)

    # reading the mask
    jpg_mask = cv2.imread(jpg_img_path)
    png_mask = cv2.imread(png_img_path)

    # analyzing jpg and png masks
    reshaped_jpg = jpg_mask.reshape((-1, jpg_mask.shape[-1]))
    reshaped_png = png_mask.reshape((-1, png_mask.shape[-1]))

    # get unique colors and count from jpg image
    jpg_unique_colors, jpg_count = np.unique(
        reshaped_jpg, 
        axis = 0, 
        return_counts = True,
        )
    
    # get unique colors and count from jpg image
    png_unique_colors, png_count = np.unique(
        reshaped_png, 
        axis = 0, 
        return_counts = True,
        )
    
    # jpg_unique_colors = get_unique_rgb_colors(jpg_mask)
    # png_unique_colors = get_unique_rgb_colors(png_mask)

    print("Jpg mask shape: ", jpg_mask.shape)
    print("Number of unique colors in JPG Mask: ", len(jpg_unique_colors))
    print("JPG Unique Colors:\n ", jpg_unique_colors)
    print("Note: Since the number of unqiue colors is large, "
          "not priting the color frequency")
    print("\n", "="*30, "\n")
    print("PNG mask Shape:  ", png_mask.shape)
    print("Number of unique colors in PNG Mask: ", len(png_unique_colors))
    print("PNG Unique Colors:\n ", png_unique_colors)
    print("PNG Color Frequency: ", png_count)
    print("\n", "="*30, "\n")

    # listing all the unique RGB labels which our PNG image or use-case has
    all_labels = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]]

    # get distortion corrected JPG image
    new_mask = get_corrected_single_channel_mask(jpg_mask, all_labels)

    # compare the unique colors and count of original PNG image and 
    # the new JPG corrected image
    reshaped_new_mask = new_mask.reshape((-1, new_mask.shape[-1]))

    # get unique colors and count from new_mask image
    new_mask_unique_colors, new_mask_count = np.unique(
        reshaped_new_mask, 
        axis = 0, 
        return_counts = True,
        )
    
    print("New mask Shape:  ", new_mask.shape)
    print("Number of unique colors in New Mask: ", len(new_mask_unique_colors))
    print("New Mask Unique Colors:\n ", new_mask_unique_colors)
    print("New Mask Color Frequency: ", new_mask_count)
    print("\n", "="*30, "\n")

    print("COMPARING PNG AND NEW MASK")
    equal_check = np.array_equal(new_mask, png_mask)
    print("Equal Status Check: ", equal_check)

    # plotting the image
    cv2.imshow("jpg-mask", jpg_mask)
    cv2.imshow("png-mask", png_mask)
    cv2.imshow("New Mask", new_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()