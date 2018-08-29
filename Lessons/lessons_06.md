# Lesson VI: Introduction to neuronal networks

### 1. Neuronal Networks: Intuition

[![Neuronal Networks: Intuition](http://img.youtube.com/vi/UKEIHK5IifI/0.jpg)](https://youtu.be/UKEIHK5IifI "Neuronal Networks: Intuition")

### 2. Introduction to Deep Learning

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 3. Starting Machine Learning

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 4. A note on Deep Learning

**A Note on Deep Learning**

The following lessons contain introductory and intermediate material on neural networks, building a neural network from 
scratch, using TensorFlow, and Convolutional Neural Networks:

* Introduction to Neural Networks
* MiniFlow
* Introduction to TensorFlow
* Deep Neural Networks
* Convolutional Neural Networks

While we highly suggest going through the all of the included content, if you already feel comfortable in any of these 
areas, feel free to skip ahead to later lessons. However, even if you have seen some of these topics before, it might be 
a good idea to get a refresher before you start working on the project!

![mini-flow](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/March/5ab2ed48_1-4-introduction-to-neural-networks2x/1-4-introduction-to-neural-networks2x.jpg)

**Already know this content in your own neural network? Feel free to skip ahead!**

### 5. Quiz: Housing prices

[![quiz housing prices](http://img.youtube.com/vi/8CSBiVKu35Q/0.jpg)](https://youtu.be/8CSBiVKu35Q "quiz housing prices")

![housing-prices](https://d17h27t6h515a5.cloudfront.net/topher/2017/November/5a0a88f8_house/house.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_01.PNG)

### 6. Solution: Housing prices

[![solution housing prices](http://img.youtube.com/vi/uhdTulw9-Nc/0.jpg)](https://youtu.be/uhdTulw9-Nc "solution housing prices")

### 7. Linear to Logistic Regression

**Linear to Logistic Regression**

Linear regression helps predict values on a continuous spectrum, like predicting what the price of a house will be.

How about classifying data among discrete classes?

Here are examples of classification tasks:

* Determining whether a patient has cancer
* Identifying the species of a fish
* Figuring out who's talking on a conference call

Classification problems are important for self-driving cars. Self-driving cars might need to classify whether an object 
crossing the road is a car, pedestrian, and a bicycle. Or they might need to identify which type of traffic sign is 
coming up, or what a stop light is indicating.

In the next video, Luis will demonstrate a classification algorithm called "logistic regression". He'll use logistic 
regression to predict whether a student will be accepted to a university.

Linear regression leads to logistic regression and ultimately neural networks, a more advanced classification tool.

### 8. Classification problems 1

**Classification Problems**

We'll start by defining what we mean by classification problems, and applying it to a simple example.

[![class prob 1](http://img.youtube.com/vi/Dh625piH7Z0/0.jpg)](https://youtu.be/Dh625piH7Z0 "class prob 1")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_02.PNG)

### 9. Classification problems 2

[![class prob 2](http://img.youtube.com/vi/46PywnGa_cQ/0.jpg)](https://youtu.be/46PywnGa_cQ "class prob 2")

### 10. Linear boundaries

[![linear boundaries](http://img.youtube.com/vi/X-uMlsBi07k/0.jpg)](https://youtu.be/X-uMlsBi07k "linear boundaries")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_03.PNG)

### 11. Higher dimensions

[![higher dimensions](http://img.youtube.com/vi/eBHunImDmWw/0.jpg)](https://youtu.be/eBHunImDmWw "higher dimensions")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_04.PNG)

### 12. Perceptrons

[![perceptrons](http://img.youtube.com/vi/hImSxZyRiOw/0.jpg)](https://youtu.be/hImSxZyRiOw "perceptrons")

_Corrections:_

* _At 3:07, the title says "Set Function". It should be "Step Function"._
* _At 3:12, the second option for y should be "0 if x<0"._

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_05.PNG)

### 13. Perceptrons II

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a49d8a_hq-perceptron/hq-perceptron.png)

###### Perceptron

Now you've seen how a simple neural network makes decisions: by taking in input data, processing that information, and 
finally, producing an output in the form of a decision! Let's take a deeper dive into the university admission example 
to learn more about processing the input data.

Data, like test scores and grades, are fed into a network of interconnected nodes. These individual nodes are [called](https://en.wikipedia.org/wiki/Perceptron) 
perceptrons, or artificial neurons, and they are the basic unit of a neural network. Each one looks at input data and 
decides how to categorize that data. In the example above, the input either passes a threshold for grades and test 
scores or doesn't, and so the two categories are: yes (passed the threshold) and no (didn't pass the threshold). These 
categories then combine to form a decision -- for example, if both nodes produce a "yes" output, then this student gains 
admission into the university.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a49d9f_hq-new-plot-perceptron-combine-v2/hq-new-plot-perceptron-combine-v2.png)

Let's zoom in even further and look at how a single perceptron processes input data.

The perceptron above is one of the two perceptrons from the video that help determine whether or not a student is 
accepted to a university. It decides whether a student's grades are high enough to be accepted to the university. You 
might be wondering: "How does it know whether grades or test scores are more important in making this acceptance 
decision?" Well, when we initialize a neural network, we don't know what information will be most important in making a 
decision. It's up to the neural network to learn for itself which data is most important and adjust how it considers 
that data.

It does this with something called **weights**.

###### Weights

When input comes into a perceptron, it gets multiplied by a weight value that is assigned to this particular input. For 
example, the perceptron above has two inputs, ```tests``` for test scores and ```grades```, so it has two associated weights that 
can be adjusted individually. These weights start out as random values, and as the neural network network learns more 
about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based 
on any errors in categorization that results from the previous weights. This is called *training* the neural network.

A higher weight means the neural network considers that input more important than other inputs, and lower weight means 
that the data is considered less important. An extreme example would be if test scores had no affect at all on 
university acceptance; then the weight of the test score input would be zero and it would have no affect on the output 
of the perceptron.

###### Summing the Input Data

Each input to a perceptron has an associated weight that represents its importance. These weights are determined during 
the learning process of a neural network, called training. In the next step, the weighted input data are summed to 
produce a single value, that will help determine the final output - whether a student is accepted to a university or 
not. Let's see a concrete example of this.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894d4d5_perceptron-graphics.001/perceptron-graphics.001.jpeg)

When writing equations related to neural networks, the weights will always be represented by some type of the letter w. 
It will usually look like a WW when it represents a matrix of weights or a ww when it represents an individual weight, 
and it may include some additional information in the form of a subscript to specify which weights (you'll see more on 
that next). But remember, when you see the letter w, think weights.

In this example, we'll use w<sub>grades</sub> for the weight of ```grades``` and w<sub></sub> test ​for the weight of ```test```.

### 14. Why "Neuronal Network" ?

[![why nn](http://img.youtube.com/vi//0.jpg)]( "why nn")

### 15. Perceptrons as logical operators

[![perceptrons as l o](http://img.youtube.com/vi//0.jpg)]( "perceptrons as l o")

### 16. Perceptrons trick

[![perceptrons trick](http://img.youtube.com/vi//0.jpg)]( "perceptrons trick")

### 17. Perceptrons algorithms

[![perceptrons algorithm](http://img.youtube.com/vi//0.jpg)]( "perceptrons algorithm")

### 18. Non-linear regions

[![NL regions](http://img.youtube.com/vi//0.jpg)]( "NL regions")

### 19. Error functions

[![err func](http://img.youtube.com/vi//0.jpg)]( "err func")

### 20. Log-loss error function

[![log loss err func](http://img.youtube.com/vi//0.jpg)]( "log loss err func")

### 21. Discrete vs Continuous

[![D vs C](http://img.youtube.com/vi//0.jpg)]( "D vs C")