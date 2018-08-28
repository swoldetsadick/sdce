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

--* Build a five steps pipeline that detects lane lines in an image (color photo)

--* Apply the pipeline to a video

--* Improve on the line extrapolation algorithm

--* Apply the new pipeline to a different video

* Reflect on your work in a written report

--* On color detection

--* On region proposal

--* Canny and Hough algorithms

--* On angle and position based outlier elimination

--* On regression

[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"

---
---

## Project

The end result of the project itself can be found at this [link](https://github.com/swoldetsadick/sdce/blob/master/Projects/01_find_lane_lines_on_the_road/CarND_LaneLines_P1/P1.ipynb) in the ipython notebook format, or [here](https://github.com/swoldetsadick/sdce/blob/master/Projects/01_find_lane_lines_on_the_road/CarND_LaneLines_P1/md/P1.md) in .md format.

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
