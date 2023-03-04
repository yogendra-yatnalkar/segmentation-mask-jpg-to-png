import numpy as np
import cv2
import os

def get_unique_rgb_colors(img):
    reshaped_img = img.reshape((-1, img.shape[-1]))
    return np.unique(reshaped_img, axis = 0)

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

# plotting the image
cv2.imshow("jpg-mask", jpg_mask)
cv2.imshow("png-mask", png_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()