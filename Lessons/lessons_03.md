# Lesson III: Computer Vision Fundamentals

### 1. Power of cameras

[![power of cameras](http://img.youtube.com/vi/lCPWJEEzUeo/0.jpg)](https://youtu.be/lCPWJEEzUeo "power of cameras")

### 2. Setting up the problem

**Finding Lane Lines on the Road**

[![finding road lane](http://img.youtube.com/vi/aIkAcXVxf2w/0.jpg)](https://youtu.be/aIkAcXVxf2w "finding road lane")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/03_01.PNG)

### 3. Color selection

**Identifying Lane Lines by Color**

[![color selection](http://img.youtube.com/vi/bNOWJ9wdmhk/0.jpg)](https://youtu.be/bNOWJ9wdmhk "color selection")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/03_02.PNG)

### 4. Color selection: Code example

**Coding up a Color Selection**

Let’s code up a simple color selection in Python.

No need to download or install anything, you can just follow along in the browser for now.

We'll be working with the same image you saw previously.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/August/57b4b3ff_test/test.jpg)

Check out the code below. First, I import _pyplot_ and _image_ from _matplotlib_. I also import _numpy_ for operating on 
the image.

`import matplotlib.pyplot as plt` <br>
`import matplotlib.image as mpimg` <br>
`import numpy as np`

I then read in an image and print out some stats. I’ll grab the x and y sizes and make a copy of the image to work with. 
NOTE: Always make a copy of arrays or other variables in Python. If instead, you say "a = b" then all changes you make 
to "a" will be reflected in "b" as well!

`# Read in the image and print out some stats` <br>
`image = mpimg.imread('test.jpg')` <br>
`print('This image is: ',type(image), 'with dimensions:', image.shape)` <br>
`# Grab the x and y size and make a copy of the image` <br>
`ysize = image.shape[0]` <br>
`xsize = image.shape[1]` <br>
`# Note: always make a copy rather than simply using "="`<br>
`color_select = np.copy(image)` <br>

Next I define a color threshold in the variables red_threshold, green_threshold, and blue_threshold and populate 
rgb_threshold with these values. This vector contains the minimum values for red, green, and blue (R,G,B) that I will 
allow in my selection.

`# Define our color selection criteria` <br>
`# Note: if you run this code, you'll find these are not sensible values!!` <br>
`# But you'll get a chance to play with them soon in a quiz` <br>
`red_threshold = 0` <br>
`green_threshold = 0` <br>
`blue_threshold = 0` <br>
`rgb_threshold = [red_threshold, green_threshold, blue_threshold]`

Next, I'll select any pixels below the threshold and set them to zero.

After that, all pixels that meet my color criterion (those above the threshold) will be retained, and those that do not 
(below the threshold) will be blacked out.

`# Identify pixels below the threshold` <br>
`thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])`<br>
`color_select[thresholds] = [0,0,0]`<br>
`# Display the image`<br>               
`plt.imshow(color_select)`<br>
`plt.show()`<br>

The result, color_select, is an image in which pixels that were above the threshold have been retained, and pixels below 
the threshold have been blacked out.

In the code snippet above, red_threshold, green_threshold and blue_threshold are all set to 0, which implies all pixels 
will be included in the selection.

In the next quiz, you will modify the values of red_threshold, green_threshold and blue_threshold until you retain as 
much of the lane lines as possible while dropping everything else. Your output image should look like the one below.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/August/57b4c566_image34/image34.png)

### 5. Quiz: Color region

In the next quiz, I want you to modify the values of the variables red_threshold, green_threshold, and blue_threshold 
until you are able to retain as much of the lane lines as possible, while getting rid of most of the other stuff. When 
you run the code in the quiz, your image will be output with an example image next to it. Tweak these variables such 
that your input image (on the left below) looks like the example image on the right.

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/test.jpg)

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/test_color_selected.jpg)

The original image (top), and color selection applied (below).

**Quiz**

`import matplotlib.pyplot as plt` <br>
`import matplotlib.image as mpimg` <br>
`import numpy as np` <br>

`# Read in the image` <br>
`image = mpimg.imread('test.jpg')` <br>

`# Grab the x and y size and make a copy of the image` <br>
`ysize = image.shape[0]` <br>
`xsize = image.shape[1]` <br>
`color_select = np.copy(image)` <br>
`# Define color selection criteria` <br>
`###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION` <br>
`red_threshold = 200 `<br>
`green_threshold = 200 `<br>
`blue_threshold = 200 `<br>
`###### `<br>

`rgb_threshold = [red_threshold, green_threshold, blue_threshold] `<br>

`# Do a boolean or with the "|" character to identify `<br>
`# pixels below the thresholds `<br>
`thresholds = (image[:,:,0] < rgb_threshold[0]) \ `<br>
            `| (image[:,:,1] < rgb_threshold[1]) \ `<br>
            `| (image[:,:,2] < rgb_threshold[2])` <br>
`color_select[thresholds] = [0,0,0]` <br>
<br>
`# Display the image` <br>                 
`plt.imshow(color_select)` <br>

![alt text](https://lh3.googleusercontent.com/WWBQqIzyzP6MHivKYReCja1v2JJIgZnktfimaQ-f7KQVyeTAKAELjYOoKrYCvudJEN6Ndu9b7NTriFzvaBA)

### 6. Region Masking

[![region masking](http://img.youtube.com/vi/ngN9Cr-QfiI/0.jpg)](https://youtu.be/ngN9Cr-QfiI "region masking")

**Coding up a Region of Interest Mask**

Awesome! Now you've seen that with a simple color selection we have managed to eliminate almost everything in the image 
except the lane lines.

At this point, however, it would still be tricky to extract the exact lines automatically, because we still have some 
other objects detected around the periphery that aren't lane lines.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/August/57b4c566_image34/image34.png)

In this case, I'll assume that the front facing camera that took the image is mounted in a fixed position on the car, 
such that the lane lines will always appear in the same general region of the image. Next, I'll take advantage of this 
by adding a criterion to only consider pixels for color selection in the region where we expect to find the lane lines.

Check out the code below. The variables left_bottom, right_bottom, and apex represent the vertices of a triangular 
region that I would like to retain for my color selection, while masking everything else out. Here I'm using a 
triangular mask to illustrate the simplest case, but later you'll use a quadrilateral, and in principle, you could use 
any polygon.

`import matplotlib.pyplot as plt` <br>
`import matplotlib.image as mpimg` <br>
`import numpy as np` <br>

`# Read in the image and print some stats` <br>
`image = mpimg.imread('test.jpg')` <br>
`print('This image is: ', type(image), 'with dimensions:', image.shape)` <br>

`# Pull out the x and y sizes and make a copy of the image` <br>
`ysize = image.shape[0]` <br>
`xsize = image.shape[1]` <br>
`region_select = np.copy(image)` <br>

`# Define a triangle region of interest ` <br>
`# Keep in mind the origin (x=0, y=0) is in the upper left in image processing` <br>
`# Note: if you run this code, you'll find these are not sensible values!!` <br>
`# But you'll get a chance to play with them soon in a quiz ` <br>
`left_bottom = [0, 539]` <br>
`right_bottom = [900, 300]` <br>
`apex = [400, 0]` <br>

`# Fit lines (y=Ax+B) to identify the  3 sided region of interest` <br>
`# np.polyfit() returns the coefficients [A, B] of the fit` <br>
`fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)` <br>
`fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)` <br>
`fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)` <br>

`# Find the region inside the lines` <br>
`XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))` <br>
`region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))` <br>

`# Color pixels red which are inside the region of interest` <br>
`region_select[region_thresholds] = [255, 0, 0]` <br>

`# Display the image` <br>
`plt.imshow(region_select)` <br>

`# uncomment if plot does not display` <br>
`# plt.show()` <br>

### 7. Color and region combined

**Combining Color and Region Selections**

Now you've seen how to mask out a region of interest in an image. Next, let's combine the mask and color selection to 
pull only the lane lines out of the image.

Check out the code below. Here we’re doing both the color and region selection steps, requiring that a pixel meet both 
the mask and color selection requirements to be retained.

`import matplotlib.pyplot as plt` <br>
`import matplotlib.image as mpimg` <br>
`import numpy as np` <br>

`# Read in the image` <br>
`image = mpimg.imread('test.jpg')` <br>

`# Grab the x and y sizes and make two copies of the image` <br>
`# With one copy we'll extract only the pixels that meet our selection,` <br>
`# then we'll paint those pixels red in the original image to see our selection` <br>
`# overlaid on the original.` <br>
`ysize = image.shape[0]` <br>
`xsize = image.shape[1]` <br>
`color_select= np.copy(image)` <br>
`line_image = np.copy(image)` <br>

`# Define our color criteria` <br>
`red_threshold = 0` <br>
`green_threshold = 0` <br>
`blue_threshold = 0` <br>
`rgb_threshold = [red_threshold, green_threshold, blue_threshold]` <br>

`# Define a triangle region of interest (Note: if you run this code,` <br>
`# Keep in mind the origin (x=0, y=0) is in the upper left in image processing` <br>
`# you'll find these are not sensible values!!` <br>
`# But you'll get a chance to play with them soon in a quiz ;)` <br>
`left_bottom = [0, 539]` <br>
`right_bottom = [900, 300]` <br>
`apex = [400, 0]` <br>

`fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)` <br>
`fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)` <br>
`fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)` <br>

`# Mask pixels below the threshold` <br>
`color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])` <br>

`# Find the region inside the lines `<br>
`XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))` <br>
`region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))` <br>
`# Mask color selection `<br>
`color_select[color_thresholds] = [0,0,0]` <br>
`# Find where image is both colored right and in the region `<br>
`line_image[~color_thresholds & region_thresholds] = [255,0,0]` <br>

`# Display our two output images` <br>
`plt.imshow(color_select)` <br>
`plt.imshow(line_image)` <br>

`# uncomment if plot does not display` <br>
`# plt.show()` <br>

In the next quiz, you can vary your color selection and the shape of your region mask (vertices of a triangle 
left_bottom, right_bottom, and apex), such that you pick out the lane lines and nothing else.

### 8. Quiz: Color - Region

In this next quiz, I've given you the values of red_threshold, green_threshold, and blue_threshold but now you need to 
modify left_bottom, right_bottom, and apex to represent the vertices of a triangle identifying the region of interest in 
the image. When you run the code in the quiz, your output result will be several images. Tweak the vertices until your 
output looks like the examples shown below.

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/test.jpg)

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/test_color_masked.jpg)

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/lines_painted.png)

The original image (top), region and color selection applied (middle) and lines identified (bottom).

**Quiz**

`import matplotlib.pyplot as plt` <br>
`import matplotlib.image as mpimg` <br>
`import numpy as np` <br>

`# Read in the image` <br>
`image = mpimg.imread('test.jpg')` <br>

`# Grab the x and y size and make a copy of the image` <br>
`ysize = image.shape[0]` <br>
`xsize = image.shape[1]` <br>
`color_select = np.copy(image)` <br>
`line_image = np.copy(image)` <br>

`# Define color selection criteria` <br>
`# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION` <br>
`red_threshold = 200` <br>
`green_threshold = 200` <br>
`blue_threshold = 200` <br>

`rgb_threshold = [red_threshold, green_threshold, blue_threshold]` <br>

`# Define the vertices of a triangular mask.` <br>
`# Keep in mind the origin (x=0, y=0) is in the upper left` <br>
`# MODIFY THESE VALUES TO ISOLATE THE REGION ` <br>
`# WHERE THE LANE LINES ARE IN THE IMAGE` <br>
`left_bottom = [100, 539]` <br>
`right_bottom = [1000, 539] `<br>
`apex = [470, 330]` <br>

`# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle `<br>
`# np.polyfit returns the coefficients [A, B] of the fit` <br>
`fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)` <br>
`fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)` <br>
`fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)` <br>

`# Mask pixels below the threshold` <br>
`color_thresholds = (image[:,:,0] < rgb_threshold[0]) | (image[:,:,1] < rgb_threshold[1]) | (image[:,:,2] < rgb_threshold[2])` <br>

`# Find the region inside the lines` <br>
`XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))` <br>
`region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]))` <br>
                    
`# Mask color and region selection` <br>
`color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]` <br>
`# Color pixels red where both color and region selections met` <br>
`line_image[~color_thresholds & region_thresholds] = [255, 0, 0]` <br>

`# Display the image and show region and color selections` <br>
`plt.imshow(image)` <br>
`x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]` <br>
`y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]` <br>
`plt.plot(x, y, 'b--', lw=4)` <br>
`plt.imshow(color_select)` <br>
`plt.imshow(line_image)` <br>

![alt text](https://lh3.googleusercontent.com/UBjTSEV9h3PBvfMGjfa20Qjd0OH0nrsuBfpy4rU5xcgjvLx3regAxIUvrHRcKS5HTavpCF0buTuKokjdirc)

### 9. Finding lines of any color

**Finding Lines of any Color**

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/August/57b4ca33_image30/image30.png)

So you found the lane lines... simple right? Now you’re ready to upload the algorithm to the car and drive autonomously 
right?? Well, not quite yet ;)

As it happens, lane lines are not always the same color, and even lines of the same color under different lighting 
conditions (day, night, etc) may fail to be detected by our simple color selection.

What we need is to take our algorithm to the next level to detect lines of any color using sophisticated computer vision 
methods.

So, what is computer vision?

### 10. What is computer vision

[![region masking](http://img.youtube.com/vi/wxQhfSdxjKU/0.jpg)](https://youtu.be/wxQhfSdxjKU "region masking")

In rest of this lesson, we’ll introduce some computer vision techniques with enough detail for you to get an intuitive 
feel for how they work.

You'll learn much more about these topics during the Computer Vision module later in the program.

We also recommend the free Udacity course, [Introduction to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810).

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/August/57b4d197_image14/image14.png)

Throughout this Nanodegree Program, we will be using Python with OpenCV for computer vision work. OpenCV stands for 
Open-Source Computer Vision. For now, you don't need to download or install anything, but later in the program we'll 
help you get these tools installed on your own computer.

OpenCV contains extensive libraries of functions that you can use. The OpenCV libraries are well documented, so if you’
re ever feeling confused about what the parameters in a particular function are doing, or anything else, you can find a 
wealth of information at [opencv.org](http://opencv.org/).

### 11. Canny edge detection

[![edge detection](http://img.youtube.com/vi/Av2GsgQWX8I/0.jpg)](https://youtu.be/Av2GsgQWX8I "edge detection")

[![edge detection](http://img.youtube.com/vi/LQM--KPJjD0/0.jpg)](https://youtu.be/LQM--KPJjD0 "edge detection")

**Note! The standard location of the origin (x=0, y=0) for images is in the top left corner with y values increasing 
downward and x increasing to the right. This might seem weird at first, but if you think about an image as a matrix, it 
makes sense that the "00" element is in the upper left.**

Now let's try a quiz. Below, I’m plotting a cross section through this image. Where are the areas in the image that are 
most likely to be identified as edges?

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58014f3a_17-q-canny-intro-quiz-2/17-q-canny-intro-quiz-2.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/03_03.PNG)

### 12. Canny to detect lane lines

**Canny Edge Detection in Action**

Now that you have a conceptual grasp on how the Canny algorithm works, it's time to use it to find the edges of the lane 
lines in an image of the road. So let's give that a try.

First, we need to read in an image:

`import matplotlib.pyplot as plt`<br>
`import matplotlib.image as mpimg`<br>
`image = mpimg.imread('exit-ramp.jpg')`<br>
`plt.imshow(image)`

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/585047e6_exit-ramp/exit-ramp.jpg)

Here we have an image of the road, and it's fairly obvious by eye where the lane lines are, but what about using 
computer vision?

Let's go ahead and convert to grayscale.

`import cv2  #bringing in OpenCV libraries` <br>
`gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion` <br>
`plt.imshow(gray, cmap='gray')`

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/September/57ccc862_grayscale/grayscale.jpg)

Let’s try our Canny edge detector on this image. This is where OpenCV gets useful. First, we'll have a look at the 
parameters for the OpenCV Canny function. You will call it like this:

`edges = cv2.Canny(gray, low_threshold, high_threshold)` <br>

In this case, you are applying Canny to the image gray and your output will be another image called edges. low_threshold 
and high_threshold are your thresholds for edge detection.

The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below 
the low_threshold. Next, pixels with values between the low_threshold and high_threshold will be included as long as 
they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges 
and black everywhere else. See the [OpenCV Canny Docs](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html) for more details.

What would make sense as a reasonable range for these parameters? In our case, converting to grayscale has left us with 
an [8-bit](https://en.wikipedia.org/wiki/8-bit) image, so each pixel can take 2^8 = 256 possible values. Hence, the pixel values range from 0 to 255.

This range implies that derivatives (essentially, the value differences from pixel to pixel) will be on the scale of 
tens or hundreds. **So, a reasonable range for your threshold parameters would also be in the tens to hundreds.**

As far as a ratio of low_threshold to high_threshold, [John Canny himself](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps) 
recommended a low to high ratio of 1:2 or 1:3.

We'll also include Gaussian smoothing, before running Canny, which is essentially a way of suppressing noise and 
spurious gradients by averaging (check out the [OpenCV docs for GaussianBlur](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur)). 
cv2.Canny() actually applies Gaussian smoothing internally, but we include it here because you can get a different 
result by applying further smoothing (and it's not a changeable parameter within cv2.Canny()!).

You can choose the kernel_size for Gaussian smoothing to be any odd number. A larger kernel_size implies averaging, or 
smoothing, over a larger area. The example in the previous lesson was kernel_size = 3.

Note: If this is all sounding complicated and new to you, don't worry! We're moving pretty fast through the material 
here, because for now we just want you to be able to use these tools. If you would like to dive into the math 
underpinning these functions, please check out the free Udacity course, [Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810), 
where the third lesson covers Gaussian filters and the sixth and seventh lessons cover edge detection.

`#doing all the relevant imports`<br>
`import matplotlib.pyplot as plt`<br>
`import matplotlib.image as mpimg`<br>
`import numpy as np`<br>
`import cv2`<br>

`# Read in the image and convert to grayscale`<br>
`image = mpimg.imread('exit-ramp.jpg')`<br>
`gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`<br>

`# Define a kernel size for Gaussian smoothing / blurring`<br>
`# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally`<br>
`kernel_size = 3`<br>
`blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)`<br>

`# Define parameters for Canny and run it`<br>
`# NOTE: if you try running this code you might want to change these!`<br>
`low_threshold = 1`<br>
`high_threshold = 10`<br>
`edges = cv2.Canny(blur_gray, low_threshold, high_threshold)`<br>

`# Display the image`<br>
`plt.imshow(edges, cmap='Greys_r')`

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/September/57ccc9f9_edges-exitramp/edges-exitramp.jpg)

Here I've called the OpenCV function Canny on a Gaussian-smoothed grayscaled image called blur_gray and detected edges 
with thresholds on the gradient of high_threshold, and low_threshold.

In the next quiz you'll get to try this on your own and mess around with the parameters for the Gaussian smoothing and 
Canny Edge Detection to optimize for detecting the lane lines and not a lot of other stuff.

### 13. Quiz: Canny edges

Now it’s your turn! Try using Canny on your own and fiddle with the parameters for the Gaussian smoothing and Edge 
Detection to optimize for detecting the lane lines well without detecting a lot of other stuff. Your result should look 
like the example shown below.

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/exit-ramp.jpg)

![alt text](https://s3.amazonaws.com/udacity-sdc/new+folder/exit_ramp_edges.jpg)

The original image (top), and edge detection applied (bottom).

`# Do all the relevant imports`<br>
`import matplotlib.pyplot as plt`<br>
`import matplotlib.image as mpimg`<br>
`import numpy as np`<br>
`import cv2`<br>

`# Read in the image and convert to grayscale`<br>
`# Note: in the previous example we were reading a .jpg `<br>
`# Here we read a .png and convert to 0,255 bytescale`<br>
`image = mpimg.imread('exit-ramp.jpg')`<br>
`gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)`<br>

`# Define a kernel size for Gaussian smoothing / blurring`<br>
`kernel_size = 3 # Must be an odd number (3, 5, 7...)`<br>
`blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)`<br>

`# Define our parameters for Canny and run it`<br>
`low_threshold = 115`<br>
`high_threshold = 125`<br>
`edges = cv2.Canny(blur_gray, low_threshold, high_threshold)`<br>

`# Display the image`<br>
`plt.imshow(edges, cmap='Greys_r')`

![alt text](https://lh3.googleusercontent.com/yYEdL7cBbgy-puiBVzkU1Jzz-gAFbCmi1qsFvlKPLGoKOS01DEg0itjaqq9Kkaid6jeZUkU24diR24jNuQ)

### 14. Hough transform
### 15. Hough transform to find lane lines
### 16. Quiz: Hough transform
