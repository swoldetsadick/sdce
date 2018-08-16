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

### 6. Power of cameras
### 7. Power of cameras
### 8. Power of cameras
### 9. Power of cameras
### 10. Power of cameras
### 11. Power of cameras
### 12. Power of cameras
### 13. Power of cameras
### 14. Power of cameras
### 15. Power of cameras
### 16. Power of cameras