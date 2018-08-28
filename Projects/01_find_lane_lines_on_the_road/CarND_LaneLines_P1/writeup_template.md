# **Finding Lane Lines on the Road** 

## Project writeup

### This project writeup concludes the first three lessons of the first term of the self-driving car nano-degree program from Udacity focused on fundamental in computer vision.

### The lessons revisit fundamental notions of computer vision, and this project makes use of Python programming language and Python modules such as scikit learn and OpenCV to propose an application of presented notions.

### Said application is focused on detecting road lane lines via classical computer vision methods.

[![Generic badge](https://img.shields.io/badge/framework-python-blue.svg)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/framework-opencv-blue.svg)](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
[![Generic badge](https://img.shields.io/badge/framework-scikit.learn-blue.svg)](http://scikit-learn.org/stable/)
[![Generic badge](https://img.shields.io/badge/domain-computer.vision-green.svg)](https://en.wikipedia.org/wiki/Computer_vision)
[![Website udacity.com](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://classroom.udacity.com/) 

---

## Project writeup

### The goals of this project are the following:

* Make a pipeline that finds lane lines on the road

    * Build a five steps pipeline that detects lane lines in an image (color photo)

    * Apply the pipeline to a video

    * Improve on the line extrapolation algorithm

    * Apply the new pipeline to a different video

* Reflect on your work in a written report

    * On color detection

    * On region proposal

    * Canny and Hough algorithms

    * On angle and position based outlier elimination

    * On regression

[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "original"
[image2]: ./examples/grayscale.jpg "proposed region"
[image3]: ./examples/grayscale.jpg "color filtering"
[image4]: ./examples/grayscale.jpg "grayscale"
[image5]: ./examples/grayscale.jpg "canny"
[image6]: ./examples/grayscale.jpg "blurred"
[image7]: ./examples/grayscale.jpg "final"

---
---

## Project

The end result of the project itself can be found at this [link](https://github.com/swoldetsadick/sdce/blob/master/Projects/01_find_lane_lines_on_the_road/CarND_LaneLines_P1/P1.ipynb) in the ipython notebook format, or [here](https://github.com/swoldetsadick/sdce/blob/master/Projects/01_find_lane_lines_on_the_road/CarND_LaneLines_P1/md/P1.md) in .md format.

## Reflection

### 1. Lane recognition pipeline

The pipeline consisted of 4 major steps. 

A. Region selection

Immediately after reading in the image itself, we apply a **region selection** algorithm to it. The selection region is 
in the form of a trapezoid shape, wider at the bottom to capture a maximum width of the image. The length of the 
trapezoid is selected so it includes the maximum length of lanes, without including far way object and/or the horizon.

![]()
_Original Image_

![]()
_Proposed region_

B. Color selection

The resulting image is then put through a color selecting algorithm. Indeed, lane lines that must be detected in this
project have white or yellow color. This is achieved by first changing the representation of the input image from RGB
to HSV (Hue, Saturation, Value) representation, that is better suited to OpenCV's color filtering algorithms from 
literature. Then two different mask one for yellow and one for white are built, combined and applied to the imput image.
After this process, only white and yellow elements, of certain HSV values, of the input image are kept.

![]()
_Color filtering_

C. Grayscale, Canny and Blurring

Immediately after color selection, images are grayscaled and a Canny edge detection algorithm is run over the resulting
black and white image. Canny is a gradient based edge detector that, very much like other first order difference based
edge detectors has two parameters to tune. We have not delved into tuning parameters here as taking threshold values
observed in lesson 3 give decent results.
The resulting edges have a lot of noise, and in order to remove and smooth lines, we use a gaussian blurring algorithm
of kernel size 11.

![]()
_Grayscaled image_

![]()
_Canny_

![]()
_Blurred image_

D. Hough lines

Once only edges are extracted from the natural images, only straight line edges are of interest of lane lines. Hence,
we apply a Hough transform algorithm to extract lines. Once again, as parameters seen in lesson 3 were sufficient enough
no formal parameter search was performed.
By the end, lines detected are superposed on the original image as faded red lines.

![]()
_Final image_

This pipeline is run on the 6 example images, then on a first video.

### 2. Improving draw_line() function

The idea is to average all Hough transform detected lines, for one lane lines, and extrapolate just that, a line. The 
first idea consisted of using, linear regression, and particularly ordinary least square to fit a line into the points
cloud formed by start and end points of detected lines.

However, this method is very much susceptible to outliers. We then used two hand made filters to exclude outliers.
The first consisted on calculating the inclination on lines. If inclination of a line was found not to be between 26 and
46 degrees in absolute value, the points were dropped as outliers.

Similarly, if x and y values of points deviated by 750 and 250 respectively from the average value of xs and ys, then 
the point was treated as being an outlier.

For the rest, a robust linear regression estimator RANSAC was used in order to lessen the impact of outliers on the 
regression. In the end x values for the minimum and maximum y values used in the pipeline (via trapezoid) is derived
from regression coefficient and intercept, and resulting line plotted on the original image.

This new improved pipeline is then run on a second video.


### 3. Shortcomings and potential improvements

* On color detection

    The extraction of yellow and white colors limits itself to a certain type of yellow and white within the spectrum.
    It is however the nature of light that under different lighting conditions, colors change. Indeed colors may vary 
    considerably depending on prevailing lighting conditions.
    
    For example, yellow, orange, and brown, are all the same “hue” (actual color), they differ only in brightness.
    Hence, it is very possible that proposed color detection may not be able to generalize.

    > An improvement proposal would be to apply grayscale on the different R, G and B channels of the original image 
    separately, in addition to Canny and blurring/smoothing algorithms then merge the results.

* On region proposal
    
    As seen above, this specific pipeline uses a trapezoid shape to propose a region of interest. This step relies on 
    one key assumption, that camera fixed on the car do not move. However hardware failure are always prepoderant as 
    shown in the "Stanley film".
    
    Again the example in this project reduce a complex reality. First, we always observe driving situation where leading 
    car is further out in the horizon, and second, the ego car itself is never between two lane lines.
    
    In case of the first, we might introduce additional complexity to the problem because leading cars would add to 
    outlier points to be filtered out later. But it would be especially problematic if said leading car is partially 
    obfuscating lane lines.
    
    In the second case, we might totally miss out in detecting lane lines.
    
    > One improvement proposal would be here to completely do away with region proposal early on, and re-introduce it 
    later-on, once some lanes are detected by specifying which region could be of interest potentially using possible 
    detected lanes.

    
* Canny and Blurring/smoothing

    First, let's note that Canny is the best edge detector out there when computation cost is taken into account. 
    Similarly, we have observed that gaussian blurring does a good enough job for thinning detected edges.
    
    However, we have not delved into exploring differing types of blurring and edge detection algorithms.
    
    > A proposition here would be to implement edge detection algorithms such as phase stretch transform and phase 
    congruency based edge detectors, or the more recent topology based edge detector.
    A go to alternative to gaussian blurring would be median blurring.
    
* On angle and position based outlier elimination

    The whole implementation of angle and position based filter assumes that the car is driving on a relatively straight
    road with minimum curvature. When road curves are more pronounced, such filter can become completely useless.

* On regression

    We have seen already that creating one consistent lane line out of the detected lane line points is problematic 
    because in case of the linear regression via OLS, robustness to outliers poses a problem.
    Hence we have introduced RANSAC estimators, that are generally median based.
    
    However the issue here is that while at the bottom of the image a number of point are detected on the line, as we go
    top on the photo the lane line thins out, and detected points become fewer. Hence, regression line tend to 
    under-estimate the coefficient of the line, regression line slightly departing the lane line on top of image.
    
    Additionally, linear regression only extrapolates line correctely when driving on a straight line. Curvatures would
    be a problem to extrapolate.
    
    > A proposal to solve the first porblem would be to create additional point on the line on top of the image if said
    line is not an outlier to compensate for bottom points over-weight.
    For taking curvature into account, use non-parametric regession methods such as splines or kernel regressions to fit
    a line onto the point cloud.
