# Segmentation Mask: The curse of JPG and a possible cure

---

- Image Segmentation task needs a set of training images and its corresponding segmentation mask. Within the AI industry, I have seen it first-hand that deveopers and companies store their self-tagged pracious segmentation labels in "JPG" format.

- **But, they often dont realize that JPG images are stored using lossy compression format and the segmentation mask gets corrupted when saved in JPG format**

- In this repo, we will see how JPG's information loss is a big problem during segmentation training and proposes a heuristic approach to convert the JPG segmentation masks to PNG masks. 

---

### JPG Corruption:

- Let's consider a RGB segmentation mask saved as JPG and PNG simultaneously. I have created one for our task which can be viewed below. This RGB mask is later saved in JPG and PNG format. 

![](D:\software-dev\jpg-to-png\screenshots\mask-creation.png)

- If both the JPG and PNG masks are zoomed to an extensive level, we can observe that the edges of JPG masks are blurry and varous shades of green are visible. On the other hand, the PNG image is intact. 

<img title="" src="file:///D:/software-dev/jpg-to-png/screenshots/jpg-curroption.png" alt="" width="696" data-align="center">

- If we look at the unique colors from the masks, we can observe that, JPG image has lot of unique colors but our mask should only contain 4 unique colors which are: RED, BLUE, GREEN and WHITE. 
  
  <mark>JPG MASK: **<u>1041 Unique colors</u>**</mark>
  
  <mark>PNG MASK: **<u>4 Unique colors</u>**</mark>
  
  ![](D:\software-dev\jpg-to-png\screenshots\unique_color_count.png)

- **If you are a deep-learning practioner, you will realize that that having so many different colors is a segmentation mask has very undesirable effect on the segmentation training.**

---

### Finally, we also propose a heuristic based solution to convert curropted JPG masks to 100% accurate PNG masks.

- The solution is available in the code-file: <mark>**jpg-to-png.py**</mark>