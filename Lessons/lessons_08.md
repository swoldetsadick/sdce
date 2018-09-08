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

![quiz 08.08](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_01.PNG)

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

![quiz 08.09](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_02.PNG)

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

### 12. Let's make a deal

[![Let's make a deal](http://img.youtube.com/vi/l1xMgirpwCU/0.jpg)](https://youtu.be/l1xMgirpwCU "Let's make a deal")

### 13. Training your logistic classifier

[![Training your logistic classifier](http://img.youtube.com/vi/WQsdr1EJgz8/0.jpg)](https://youtu.be/WQsdr1EJgz8 "Training your logistic classifier")

### 14. TensorFlow linea function

###### TensorFlow Linear Function

Let’s derive the function ```y = Wx + b```. We want to translate our input, ```x```, to labels, ```y```.

For example, imagine we want to classify images as digits.

```x``` would be our list of pixel values, and ```y``` would be the logits, one for each digit. Let's take a look at 
```y = Wx```, where the weights, ```W```, determine the influence of ```x``` at predicting each ```y```.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af21f64_wx-1/wx-1.jpg)

```y``` = Wx allows us to segment the data into their respective labels using a line.

However, this line has to pass through the origin, because whenever ```x``` equals 0, then ```y``` is also going to 
equal 0.

We want the ability to shift the line away from the origin to fit more complex data. The simplest solution is to add a 
number to the function, which we call “bias”.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af21f8c_wx-b/wx-b.jpg)

Our new function becomes ```Wx + b```, allowing us to create predictions on linearly separable data. Let’s use a 
concrete example and calculate the logits.

###### Matrix Multiplication Quiz

Calculate the logits ```a``` and ```b``` for the following formula.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af21fd9_codecogseqn-13/codecogseqn-13.gif)

![quiz 08.14](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_03.PNG)

![quiz 08.14](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_04.PNG)

###### Transposition

We've been using the ```y = Wx + b``` function for our linear function.

But there's another function that does the same thing, ```y = xW + b```. These functions do the same thing and are 
interchangeable, except for the dimensions of the matrices involved.

To shift from one function to the other, you simply have to swap the row and column dimensions of each matrix. This is 
called transposition.

For rest of this lesson, we actually use ```xW + b```, because this is what TensorFlow uses.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af220e2_codecogseqn-18/codecogseqn-18.gif)

The above example is identical to the quiz you just completed, except that the matrices are transposed.

```x``` now has the dimensions 1x3, ```W``` now has the dimensions 3x2, and ```b``` now has the dimensions 1x2. 
Calculating this will produce a matrix with the dimension of 1x2.

You'll notice that the elements in this 1x2 matrix are the same as the elements in the 2x1 matrix from the quiz. Again, 
these matrices are simply transposed.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af2210f_codecogseqn-20/codecogseqn-20.gif)

We now have our logits! The columns represent the logits for our two labels.

Now you can learn how to train this function in TensorFlow.

###### Weights and Bias in TensorFlow

The goal of training a neural network is to modify weights and biases to best predict the labels. In order to use 
weights and bias, you'll need a Tensor that can be modified. This leaves out ```tf.placeholder()``` and ```tf.constant()```
, since those Tensors can't be modified. This is where ```tf.Variable``` class comes in.

###### tf.Variable()

```
x = tf.Variable(5)
```

The ```tf.Variable``` class creates a tensor with an initial value that can be modified, much like a normal Python variable. 
This tensor stores its state in the session, so you must initialize the state of the tensor manually. You'll use the 
```tf.global_variables_initializer()``` function to initialize the state of all the Variable tensors.

###### Initialization

```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

The ```tf.global_variables_initializer()``` call returns an operation that will initialize all TensorFlow variables from 
the graph. You call the operation using a session to initialize all the variables as shown above. Using the 
```tf.Variable``` class allows us to change the weights and bias, but an initial value needs to be chosen.

Initializing the weights with random numbers from a normal distribution is good practice. Randomizing the weights helps 
the model from becoming stuck in the same place every time you train it. You'll learn more about this in the next lesson
, when you study gradient descent.

Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. You'll 
use the ```tf.truncated_normal()``` function to generate random numbers from a normal distribution.

###### tf.truncated_normal()

```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

The ```tf.truncated_normal()``` function returns a tensor with random values from a normal distribution whose magnitude 
is no more than 2 standard deviations from the mean.

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's 
use the simplest solution, setting the bias to 0.

```
tf.zeros()
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

The ```tf.zeros()``` function returns a tensor with all zeros.

###### Linear Classifier Quiz

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af2214a_mnist-012/mnist-012.png)

You'll be classifying the handwritten numbers ```0```, ```1```, and ```2``` from the MNIST dataset using TensorFlow. The 
above is a small sample of the data you'll be training on. Notice how some of the 1s are written with a [serif](https://en.wikipedia.org/wiki/Serif) 
at the top and at different angles. The similarities and differences will play a part in shaping the weights of the model.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af22171_weights-0-1-2/weights-0-1-2.png)

The images above are trained weights for each label (```0```, ```1```, and ```2```). The weights display the unique 
properties of each digit they have found. Complete this quiz to train your own weights using the MNIST dataset.

###### Instructions

1. Open quiz.py.
    1. Implement ```get_weights``` to return a ```tf.Variable``` of weights
    2. Implement ```get_biases``` to return a ```tf.Variable``` of biases
    3. Implement ```xW + b``` in the ```linear``` function

2. Open sandbox.py
    1. Initialize all weights

Since ```xW``` in ```xW + b``` is matrix multiplication, you have to use the ```tf.matmul()``` function instead of 
```tf.multiply()```. Don't forget that order matters in matrix multiplication, so ```tf.matmul(a,b)``` is not the same 
as ```tf.matmul(b,a)```.

### 15. Quiz: Linear function

[Quiz here](https://github.com/swoldetsadick/sdce/blob/master/Notebooks/Lesson_8/03/quiz.md)

[Solution here](https://github.com/swoldetsadick/sdce/blob/master/Notebooks/Lesson_8/03/quiz_solution.md)

### 16. Linear update

You can’t train a neural network on a single sample. Let’s apply n samples of ```x``` to the function ```y = Wx + b```, 
which becomes ```Y = WX + B```.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583a35b9_new-wx-b-2/new-wx-b-2.jpg)

For every sample of ```X``` (```X1```, ```X2```, ```X3```), we get logits for label 1 (```Y1```) and label 2 (```Y2```).

In order to add the bias to the product of ```WX```, we had to turn ```b``` into a matrix of the same shape. This is a 
bit unnecessary, since the bias is only two numbers. It should really be a vector.

We can take advantage of an operation called broadcasting used in TensorFlow and Numpy. This operation allows arrays of 
different dimension to be multiplied with each other. For example:

```
import numpy as np
t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
u = np.array([1, 2, 3])
print(t + u)
```

The code above will print...

```
[[ 2  4  6]
 [ 5  7  9]
 [ 8 10 12]
 [11 13 15]]
```

This is because ```u``` is the same dimension as the last dimension in ```t```.

### 17. Quiz: Softmax

![Quiz: Softmax](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59a3b336_softmax/softmax.png)

###### Softmax

Congratulations on successfully implementing a linear function that outputs logits. You're one step closer to a working 
classifier.

The next step is to assign a probability to each label, which you can then use to classify the data. Use the softmax 
function to turn your logits into probabilities.

We can do this by using the formula above, which uses the input of y values and the mathematical constant "e" which is 
approximately equal to 2.718. By taking "e" to the power of any real value we always get back a positive value, this 
then helps us scale when having negative y values. The summation symbol on the bottom of the divisor indicates that we 
add together all the e^(input y value) elements in order to get our calculated probability outputs.

###### Quiz

For the next quiz, you'll implement a ```softmax(x)``` function that takes in x, a one or two dimensional array of logits.

In the one dimensional case, the array is just a single set of logits. In the two dimensional case, each column in the 
array is a set of logits. The ```softmax(x)``` function should return a NumPy array of the same shape as x.

For example, given a one-dimensional array:

```
# logits is a one-dimensional array with 3 elements
logits = [1.0, 2.0, 3.0]
# softmax will return a one-dimensional array with 3 elements
print softmax(logits)
```

```
$ [ 0.09003057  0.24472847  0.66524096]
```

Given a two-dimensional array where each column represents a set of logits:

```
# logits is a two-dimensional array
logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
# softmax will return a two-dimensional array with the same shape
print softmax(logits)
```

```
$ [
    [ 0.09003057  0.00242826  0.01587624  0.33333333]
    [ 0.24472847  0.01794253  0.11731043  0.33333333]
    [ 0.66524096  0.97962921  0.86681333  0.33333333]
  ]
```

Implement the softmax function, which is specified by the formula at the top of the page.

The probabilities for each column must sum to 1. Feel free to test your function with the inputs above.

_main-code.py_
````python
# Solution is available in the other "solution.py" tab
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    exp = np.exp(x)
    try:
        _ = len(exp[0])
        return exp/exp.sum(axis=0, keepdims=True)
    except:
        return exp/np.sum(exp)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
````

_solution.py_
````python
# Quiz Solution
# Note: You can't run code in this tab
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
````

![quiz 08.14](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_05.PNG)

### 18. Quiz: TensorFlow softmax workspace

###### TensorFlow softmax

Now that you've built a softmax function from scratch, let's see how softmax is done in TensorFlow.

```
x = tf.nn.softmax([2.0, 1.0, 0.2])
```

Easy as that! ```tf.nn.softmax()``` implements the softmax function for you. It takes in logits and returns softmax 
activations.

###### Quiz

Use the softmax function in the quiz below to return the softmax of the logits.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2017/February/58950908_softmax-input-output/softmax-input-output.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_06.PNG)

###### Quiz

Answer the following 2 questions about softmax.

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_07.PNG)

### 19. One-hot encoding

[![One-hot encoding](http://img.youtube.com/vi/phYsxqlilUk/0.jpg)](https://youtu.be/phYsxqlilUk "One-hot encoding")

### 20. Quizz: One-hot encoding

![quiz 8.20](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58015400_base-hot-enc/base-hot-enc.png)

![quiz 08.14](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_08.PNG)

### 21. Cross entropy

[![Cross entropy](http://img.youtube.com/vi/tRsSi_sqXjI/0.jpg)](https://youtu.be/tRsSi_sqXjI "Cross entropy")

### 22. Minimizing cross-entropy

[![Minimizing cross-entropy](http://img.youtube.com/vi/YrDMXFhvh9E/0.jpg)](https://youtu.be/YrDMXFhvh9E "Minimizing cross-entropy")

### 23. Practical aspects of learning

[![Practical aspects of learning](http://img.youtube.com/vi/bKqkRFOOKoA/0.jpg)](https://youtu.be/bKqkRFOOKoA "Practical aspects of learning")

### 24. Quiz: Numerical stability

[![Quiz: Numerical stability](http://img.youtube.com/vi/_SbGcOS-jcQ/0.jpg)](https://youtu.be/_SbGcOS-jcQ "Quiz: Numerical stability")

![quiz 08.14](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_09.PNG)

### 25. Normalized inputs and initial weights

[![Normalized inputs and initial weights](http://img.youtube.com/vi/WaHQ9-UXIIg/0.jpg)](https://youtu.be/WaHQ9-UXIIg "Normalized inputs and initial weights")

### 26. Measuring performance

[![Measuring performance](http://img.youtube.com/vi/byP0DJImOSk/0.jpg)](https://youtu.be/byP0DJImOSk "Measuring performance")

### 27. Transition: Overfitting ---> Dataset size

[![Transition: Overfitting ---> Dataset size](http://img.youtube.com/vi/Lmxem7ud9yk/0.jpg)](https://youtu.be/Lmxem7ud9yk "Transition: Overfitting ---> Dataset size")

### 28. Validation and test set size

[![Validation and test set size](http://img.youtube.com/vi/iC2QOiavbrw/0.jpg)](https://youtu.be/iC2QOiavbrw "Validation and test set size")

### 29. Quiz: Validation set size

[![Quiz: Validation set size](http://img.youtube.com/vi/-2XvoG6WD9k/0.jpg)](https://youtu.be/-2XvoG6WD9k "Quiz: Validation set size")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_10.PNG)

### 30. Validation test set size continued

[![Validation test set size continued](http://img.youtube.com/vi/cgoB-MJObmw/0.jpg)](https://youtu.be/cgoB-MJObmw "Validation test set size continued")

### 31. Optimizing a logistic classifier

[![Optimizing a logistic classifier](http://img.youtube.com/vi/U_7nO1dm2tY/0.jpg)](https://youtu.be/U_7nO1dm2tY "Optimizing a logistic classifier")

### 32. Stochastic gradient descent

[![Stochastic gradient descent](http://img.youtube.com/vi/U9iEGUd9kJ0/0.jpg)](https://youtu.be/U9iEGUd9kJ0 "Stochastic gradient descent")

### 33. Momentum and learning rate decay

[![Momentum and learning rate decay](http://img.youtube.com/vi/O3QYdmQjXds/0.jpg)](https://youtu.be/O3QYdmQjXds "Momentum and learning rate decay")

### 34. Parameter hyperspace

[![Parameter hyperspace](http://img.youtube.com/vi/5a3-iIhdguc/0.jpg)](https://youtu.be/5a3-iIhdguc "Parameter hyperspace")

### 35. Quiz: Mini-batch

###### Mini-batching

In this section, you'll go over what mini-batching is and how to apply it in TensorFlow.

Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides 
the ability to train a model, even if a computer lacks the memory to store the entire dataset.

Mini-batching is computationally inefficient, since you can't calculate the loss simultaneously across all samples. 
However, this is a small price to pay in order to be able to run the model at all.

It's also quite useful combined with SGD. The idea is to randomly shuffle the data at the start of each epoch, then 
create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches 
are random, you're performing SGD with each batch.

Let's look at the MNIST dataset with weights and a bias to see if your machine can handle it.

```
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
```

###### Question 1

Calculate the memory size of ```train_features```, ```train_labels```, ```weights```, and ```bias``` in bytes. Ignore 
memory for overhead, just calculate the memory required for the stored data.

You may have to look up how much memory a float32 requires, using [this link](https://en.wikipedia.org/wiki/Single-precision_floating-point_format).

_train_features Shape: (55000, 784) Type: float32_

_train_labels Shape: (55000, 10) Type: float32_

_weights Shape: (784, 10) Type: float32_

_bias Shape: (10,) Type: float32_

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_11.PNG)

The total memory space required for the inputs, weights and bias is around 174 megabytes, which isn't that much memory. 
You could train this whole dataset on most CPUs and GPUs.

But larger datasets that you'll use in the future measured in gigabytes or more. It's possible to purchase more memory, 
but it's expensive. A Titan X GPU with 12 GB of memory costs over $1,000.

Instead, in order to run large models on your machine, you'll learn how to use mini-batching.

Let's look at how you implement mini-batching in TensorFlow.

###### TensorFlow Mini-batching

In order to use mini-batching, you must first divide your data into batches.

Unfortunately, it's sometimes impossible to divide the data into batches of exactly equal size. For example, imagine you
'd like to create batches of 128 samples each from a dataset of 1000 samples. Since 128 does not evenly divide into 1000
, you'd wind up with 7 batches of 128 samples, and 1 batch of 104 samples. (7*128 + 1*104 = 1000)

In that case, the size of the batches would vary, so you need to take advantage of TensorFlow's ```tf.placeholder()``` 
function to receive the varying batch sizes.

Continuing the example, if each sample had ```n_input = 784``` features and ```n_classes = 10``` possible labels, the 
dimensions for ```features``` would be ```[None, n_input]``` and ```labels``` would be ```[None, n_classes]```.

```
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```

What does ```None``` do here?

The ```None``` dimension is a placeholder for the batch size. At runtime, TensorFlow will accept any batch size greater 
than 0.

Going back to our earlier example, this setup allows you to feed ```features``` and ```labels``` into the model as 
either the batches of 128 samples or the single batch of 104 samples.

###### Question 2

Use the parameters below, how many batches are there, and what is the last batch size?

_features is (50000, 400)_

_labels is (50000, 10)_

_batch_size is 128_

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_12.PNG)

Now that you know the basics, let's learn how to implement mini-batching.

###### Question 3

Implement the ```batches``` function to batch ```features``` and ```labels```. The function should return each batch 
with a maximum size of ```batch_size```. To help you with the quiz, look at the following example output of a working 
```batches``` function.

```
# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

example_batches = batches(3, example_features, example_labels)
```

The ```example_batches``` variable would be the following:

```
[
    # 2 batches:
    #   First is a batch of size 3.
    #   Second is a batch of size 1
    [
        # First Batch is size 3
        [
            # 3 samples of features.
            # There are 4 features per sample.
            ['F11', 'F12', 'F13', 'F14'],
            ['F21', 'F22', 'F23', 'F24'],
            ['F31', 'F32', 'F33', 'F34']
        ], [
            # 3 samples of labels.
            # There are 2 labels per sample.
            ['L11', 'L12'],
            ['L21', 'L22'],
            ['L31', 'L32']
        ]
    ], [
        # Second Batch is size 1.
        # Since batch size is 3, there is only one sample left from the 4 samples.
        [
            # 1 sample of features.
            ['F41', 'F42', 'F43', 'F44']
        ], [
            # 1 sample of labels.
            ['L41', 'L42']
        ]
    ]
]
```

Implement the ```batches``` function in the "quiz.py" file below.

_sandbox.py_
````python
from quiz import batches
from pprint import pprint

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))
````

_quiz.py_
````python
import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    from random import randrange
    f = []
    while features:
        batch_feature = []
        batch_label = []
        i = 0
        if len(features) >= batch_size:
            while i <= (batch_size - 1):
                random_index = 0  # randrange(len(features))
                batch_feature.append(features.pop(random_index))
                batch_label.append(labels.pop(random_index))
                i += 1
            f.append([batch_feature, batch_label])
        else:
            break
    if features:
        f.append([features, labels])
    return f
````

_quiz_solution.py_
````python
import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_13.PNG)

Let's use mini-batching to feed batches of MNIST features and labels into a linear model.

Set the batch size and run the optimizer over all the batches with the ```batches``` function. The recommended batch 
size is 128. If you have memory restrictions, feel free to make it smaller.

_quiz.py_
````python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Set batch size
batch_size = 128
assert batch_size is not None, 'You must set the batch size'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # TODO: Train optimizer on all batches
    for batch_features, batch_labels  in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
````

_helper.py_
````python
import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
        
    return outout_batches
````

_quiz_solution.py_
````python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Set batch size
batch_size = 128
assert batch_size is not None, 'You must set the batch size'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # TODO: Train optimizer on all batches
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_14.PNG)

The accuracy is low, but you probably know that you could train on the dataset more than once. You can train a model 
using the dataset multiple times. You'll go over this subject in the next section where we talk about "epochs".

### 36. Quiz: Mini-batch 2

Let's use mini-batching to feed batches of MNIST features and labels into a linear model.

Set the batch size and run the optimizer over all the batches with the batches function. The recommended ```batch``` 
size is 128. If you have memory restrictions, feel free to make it smaller.

**This quiz is not graded, see the solution notebook for one way to solve this quiz.**

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/08_15.PNG)

[Full view here](https://github.com/swoldetsadick/sdce/blob/master/Notebooks/Lesson_8/04/quiz.md)

The accuracy is low, but you probably know that you could train on the dataset more than once. You can train a model 
using the dataset multiple times. You'll go over this subject in the next section where we talk about "epochs".


### 37. Epochs

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 38. AWS GPU instances

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 39. Introduction to Tensorflow neuronal network

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 40. Lab: Neural network workspace

[![](http://img.youtube.com/vi//0.jpg)]( "")