# Lesson VIII: Introduction to Tensorflow

### 1. Deep learning frameworks

[![DL frameworks](http://img.youtube.com/vi/Fw6cM2mpfcs/0.jpg)](https://youtu.be/Fw6cM2mpfcs "DL frameworks")

### 2. Deep learning frameworks

[![intro to DL networks](http://img.youtube.com/vi/7xRwuECaXBs/0.jpg)](https://youtu.be/7xRwuECaXBs "intro to DL networks")

### 3. What is deep learning?

[![What is deep learning?](http://img.youtube.com/vi/INt1nULYPak/0.jpg)](https://youtu.be/INt1nULYPak "What is deep learning?")

### 4. Solving problems - Big and small

[![Solving problems - Big and small](http://img.youtube.com/vi/WHcRQMGSbqg/0.jpg)](https://youtu.be/WHcRQMGSbqg "Solving problems - Big and small")

### 5. Let's get started

[![Let's get started](http://img.youtube.com/vi/ySIDqaXLhHw/0.jpg)](https://youtu.be/ySIDqaXLhHw "Let's get started")

### 6. Installing Tensorflow

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58116cd8_maxresdefault/maxresdefault.jpg)

Throughout this lesson, you'll apply your knowledge of neural networks on real datasets using [TensorFlow](https://www.tensorflow.org/)
([link for China](http://www.tensorfly.cn/)), an open source Deep Learning library created by Google.

You’ll use TensorFlow to classify images from the notMNIST dataset - a dataset of images of English letters from A to J. 
You can see a few example images below.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58051e40_notmnist/notmnist.png)

Your goal is to automatically detect the letter based on the image in the dataset. You’ll be working on your own 
computer for this lab, so, first things first, install TensorFlow!

###### Install

###### OS X, Linux, Windows

###### Prerequisites

Intro to TensorFlow requires [Python 3.4 or higher](https://www.python.org/downloads/) and [Anaconda](https://www.continuum.io/downloads). 
If you don't meet all of these requirements, please install the appropriate package(s).

###### Install TensorFlow

You're going to use an Anaconda environment for this class. If you're unfamiliar with Anaconda environments, check out 
the [official documentation](http://conda.pydata.org/docs/using/envs.html). More information, tips, and troubleshooting 
for installing tensorflow on Windows can be found [here](https://www.tensorflow.org/install/install_windows).

**Note**: If you've already created the environment for Term 1, you shouldn't need to do so again here!

Run the following commands to setup your environment:

```
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow
```

That's it! You have a working environment with TensorFlow. Test it out with the code in the _Hello, world!_ section below.

###### Docker on Windows

Docker instructions were offered prior to the availability of a stable Windows installation via pip or Anaconda. Please 
try Anaconda first, Docker instructions have been retained as an alternative to an installation via Anaconda.

###### Install Docker

Download and install Docker from the [official Docker website](https://docs.docker.com/engine/installation/windows/).

###### Run the Docker Container

Run the command below to start a jupyter notebook server with TensorFlow:

```
docker run -it -p 8888:8888 -p 6006:6006 -v `pwd`:/abs/path/to/some/local/folder tensorflow/tensorflow
```

Users in China should use the ```b.gcr.io/tensorflow/tensorflow``` instead of ```gcr.io/tensorflow/tensorflow```

You can access the jupyter notebook at localhost:8888. The server includes 3 examples of TensorFlow notebooks, but you 
can create a new notebook to test all your code.

###### Hello, world!

Try running the following code in your Python console to make sure you have TensorFlow properly installed. The console 
will print "Hello, world!" if TensorFlow is installed. Don’t worry about understanding what it does. You’ll learn about 
it in the next section.

```
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

###### Errors

If you're getting the error ```tensorflow.python.framework.errors.InvalidArgumentError: Placeholder:0 is both fed and 
fetched```, you're running an older version of TensorFlow. Uninstall TensorFlow, and reinstall it using the instructions 
above. For more solutions, check out the [Common Problems](https://www.tensorflow.org/get_started/os_setup#common_problems) section.

### 7. Hello, Tensor world!

###### Hello, Tensor World!

Let’s analyze the Hello World script you ran. For reference, I’ve added the code below.

```
import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

###### Tensor

In TensorFlow, data isn’t stored as integers, floats, or strings. These values are encapsulated in an object called a 
tensor. In the case of ```hello_constant = tf.constant('Hello World!')```, hello_constant is a 0-dimensional string 
tensor, but tensors come in a variety of sizes as shown below:

```
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

```tf.constant()``` is one of many TensorFlow operations you will use in this lesson. The tensor returned by 
```tf.constant()``` is called a constant tensor, because the value of the tensor never changes.

###### Session

TensorFlow’s api is built around the idea of a computational graph, a way of visualizing a mathematical process which 
you learned about in the MiniFlow lesson. Let’s take the TensorFlow code you ran and turn that into a graph:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580feadb_session/session.png)

A "TensorFlow Session", as shown above, is an environment for running a graph. The session is in charge of allocating 
the operations to GPU(s) and/or CPU(s), including remote machines. Let’s see how you use it.

```
with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)
```

The code has already created the tensor, ```hello_constant```, from the previous lines. The next step is to evaluate the 
tensor in a session.

The code creates a session instance, ```sess```, using ```tf.Session```. The ```sess.run()``` function then evaluates 
the tensor and returns the results.

After you run the above, you will see the following printed out:

```
'Hello World!'
```

### 8. Quiz: Tensorflow Input

###### Input

In the last section, you passed a tensor into a session and it returned the result. What if you want to use a 
non-constant? This is where ```tf.placeholder()``` and ```feed_dict``` come into place. In this section, you'll go over 
the basics of feeding data into TensorFlow.

###### tf.placeholder()

Sadly you can’t just set ```x``` to your dataset and put it in TensorFlow, because over time you'll want your TensorFlow 
model to take in different datasets with different parameters. You need ```tf.placeholder()```!

```tf.placeholder()``` returns a tensor that gets its value from data passed to the ```tf.session.run()``` function, 
allowing you to set the input right before the session runs.

###### Session’s feed_dict

```
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

Use the ```feed_dict``` parameter in ```tf.session.run()``` to set the placeholder tensor. The above example shows the 
tensor ```x``` being set to the string ```"Hello, world"```. It's also possible to set more than one tensor using 
```feed_dict``` as shown below.

```
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

**Note**: If the data passed to the ```feed_dict``` doesn’t match the tensor type and can’t be cast into the tensor type
, you’ll get the error ```“ValueError: invalid literal for...”```.

**Quiz**

Let's see how well you understand ```tf.placeholder()``` and ```feed_dict```. The code below throws an error, but I want 
you to make it return the number ```123```. Change line 11, so that the code returns the number ```123```.

### 9. Quiz: Tensorflow math

###### TensorFlow Math

Getting the input is great, but now you need to use it. You're going to use basic math functions that everyone knows and 
loves - add, subtract, multiply, and divide - with tensors. (There's many more math functions you can check out in the 
[documentation](https://www.tensorflow.org/api_docs/python/math_ops/).)

###### Addition

```
x = tf.add(5, 2)  # 7
```

You’ll start with the add function. The ```tf.add()``` function does exactly what you expect it to do. It takes in two 
numbers, two tensors, or one of each, and returns their sum as a tensor.

###### Subtraction and Multiplication

Here’s an example with subtraction and multiplication.

```
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
```

The ```x``` tensor will evaluate to ```6```, because ```10 - 4 = 6```. The ```y``` tensor will evaluate to ```10```, 
because ```2 * 5 = 10```. That was easy!

###### Converting types

It may be necessary to convert between types to make certain operators work together. For example, if you tried the 
following, it would fail with an exception:

```
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: 
```

That's because the constant 1 is an integer but the constant ```2.0``` is a floating point value and subtract expects 
them to match.

In cases like these, you can either make sure your data is all of the same type, or you can cast a value to another type
. In this case, converting the 2.0 to an integer before subtracting, like so, will give the correct result:

```
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```

###### Quiz

Let's apply what you learned to convert an algorithm to TensorFlow. The code below is a simple algorithm using division 
and subtraction. Convert the following algorithm in regular Python to TensorFlow and print the results of the session. 
You can use ```tf.constant()``` for the values ```10```, ```2```, and ```1```.

### 10. Transition to classification

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a4fb36_06-l-supervised-classification-391-1/06-l-supervised-classification-391-1.jpg)

Good job! You've accomplished a lot. In particular, you did the following:

* Ran operations in ```tf.session```.
* Created a constant tensor with ```tf.constant()```.
* Used ```tf.placeholder()``` and ```feed_dict``` to get input.
* Applied the ```tf.add()```, ```tf.subtract()```, ```tf.multiply()```, and ```tf.divide()``` functions using numeric 
data.

You know the basics of TensorFlow, so let's take a break and get back to the theory of neural networks. In the next few 
videos, you're going to learn about one of the most popular applications of neural networks - classification.

### 11. Supervised classification

[![Supervised classification](http://img.youtube.com/vi/XTGsutypAPE/0.jpg)](https://youtu.be/XTGsutypAPE "Supervised classification")


### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")