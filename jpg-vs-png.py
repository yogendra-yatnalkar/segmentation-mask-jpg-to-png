import numpy as np
import cv2
import os

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

    # plotting the image
    cv2.imshow("jpg-mask", jpg_mask)
    cv2.imshow("png-mask", png_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()