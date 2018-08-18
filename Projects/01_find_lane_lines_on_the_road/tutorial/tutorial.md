
### Some OpenCV functions: A tutorial

https://pythonprogramming.net/loading-images-python-opencv-tutorial/

#### A. cv2.inRange() and cv2.bitwise_and()


```python
# Importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```


```python
def detect_yellow_and_white(path, yellow):
    """
    This function detects yellow and white colors in images.
    """
    ## Read Image in BGR
    img = cv2.imread(path)
    ## Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## Mask for white (0, 0, 150) ~ (255, 50, 255)
    mask = cv2.inRange(hsv, (0, 0, 150), (255, 50, 255))
    if yellow:
        ## Mask for yellow (15,0,0) ~ (36, 255, 255)
        mask_yellow = cv2.inRange(hsv, (15,0,0), (36, 255, 255))
        ## Final mask for yellow and white
        mask = cv2.bitwise_or(mask, mask_yellow) 
    return cv2.bitwise_and(img, img, mask=mask)
```


```python
plt.imshow(detect_yellow_and_white("test_images/solidYellowCurve.jpg", True))
```




    <matplotlib.image.AxesImage at 0x7f53741b60b8>




![png](output_4_1.png)



```python
plt.imshow(detect_yellow_and_white("test_images/solidYellowCurve.jpg", False))
```




    <matplotlib.image.AxesImage at 0x7f537414ddd8>




![png](output_5_1.png)



```python
plt.imshow(detect_yellow_and_white("test_images/solidWhiteCurve.jpg", True))
```




    <matplotlib.image.AxesImage at 0x7f53740bf048>




![png](output_6_1.png)



```python
plt.imshow(detect_yellow_and_white("test_images/solidWhiteCurve.jpg", False))
```




    <matplotlib.image.AxesImage at 0x7f53740205f8>




![png](output_7_1.png)



```python
import os
for path in os.listdir("test_images/"):
    print("Figure {}".format(path))
    path = "test_images/" + path
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(detect_yellow_and_white(path, True))
    plt.show()
    
```

    Figure solidWhiteCurve.jpg



![png](output_8_1.png)



![png](output_8_2.png)


    Figure solidWhiteRight.jpg



![png](output_8_4.png)



![png](output_8_5.png)


    Figure solidYellowCurve2.jpg



![png](output_8_7.png)



![png](output_8_8.png)


    Figure solidYellowLeft.jpg



![png](output_8_10.png)



![png](output_8_11.png)


    Figure whiteCarLaneSwitch.jpg



![png](output_8_13.png)



![png](output_8_14.png)


    Figure solidYellowCurve.jpg



![png](output_8_16.png)



![png](output_8_17.png)


#### B. Drawing on images


```python
# Read Image
img = cv2.imread('test_images/solidWhiteCurve.jpg', cv2.IMREAD_COLOR)
# Set up line
line_point_0 = (0, 0)
line_point_1 = (150, 150)
line_color = (0, 0, 255)
line_width_in_pixels = 5
# Draw line
cv2.line(img, line_point_0, line_point_1, line_color, line_width_in_pixels) # In place
# Display image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7f5373fa1160>




![png](output_10_1.png)



```python
# Read Image
img = cv2.cvtColor(cv2.imread('test_images/solidWhiteCurve.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# Rectangle figure on image
pts = np.array([[(10, 59), (800, 59), (800, 250), (10, 250)]], np.int32)
# Figure color in rgb
color = (255, 0, 0)
# Plot figure on image
plt.imshow(cv2.fillPoly(img, pts, color))
```




    <matplotlib.image.AxesImage at 0x7f536eedac88>




![png](output_11_1.png)


#### C. Operations on images


```python
# Read Image
img = cv2.imread('test_images/solidWhiteCurve.jpg', cv2.IMREAD_COLOR)
print("Original")
plt.imshow(img)
plt.show()
# overlay two images
print("Blended")
img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
img2 = cv2.imread('test_images/solidYellowCurve.jpg')
blend = cv2.cvtColor(cv2.addWeighted(img1, 0.5, img2, 0.5, 0), cv2.COLOR_BGR2RGB)
plt.imshow(blend)
plt.show()
# To gray color
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray")
plt.imshow(gray, cmap='gray')
plt.show()
# To RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("RGB")
plt.imshow(rgb)
plt.show()
# Writing output
path = "blended.jpg"
cv2.imwrite(path, blend)
```

    Original



![png](output_13_1.png)


    Blended



![png](output_13_3.png)


    Gray



![png](output_13_5.png)


    RGB



![png](output_13_7.png)





    True


