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

When writing equations related to neural networks, the weights will always be represented by some type of the letter **w**. 
It will usually look like a _W_ when it represents a **matrix** of weights or a &#969; when it represents an **individual** weight, 
and it may include some additional information in the form of a subscript to specify which weights (you'll see more on 
that next). But remember, when you see the letter **w**, think **weights**.

In this example, we'll use w<sub>grades</sub> for the weight of ```grades``` and w<sub>test</sub> for the weight of ```test```.

For the image above, let's say that the weights are: 

w<sub>grades</sub> = -1, w<sub>test</sub> = -0.2. You don't have to be concerned with the actual values, but their 
relative values are important. w<sub>grades</sub> grades is 5 times larger than w<sub>test</sub>, which means the neural 
network considers ```grades``` input 5 times more important than ```test``` in determining whether a student will be 
accepted into a university.

The perceptron applies these weights to the inputs and sums them in a process known as **linear combination**. In our case, 
this looks like:

w<sub>grades</sub> &#8901; x<sub>grades</sub> + w<sub>test</sub> &#8901; x<sub>test</sub> = -1 &#8901; x<sub>grades</sub> - 0.2 &#8901; x<sub>test</sub> 

Now, to make our equation less wordy, let's replace the explicit names with numbers. Let's use 11 for _grades_ and 22 
for _tests_. So now our equation becomes:

w<sub>1</sub> &#8901; x<sub>1</sub> + w<sub>2</sub> &#8901; x<sub>2</sub>

In this example, we just have 2 simple inputs: grades and tests. Let's imagine we instead had m different inputs and we 
labeled them x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>m</sub>. Let's also say that the weight corresponding to x<sub>1</sub> 
is w<sub>1</sub> and so on. In that case, we would express the linear combination succintly as:

&#8721;<sub>_i_=1</sub><sup>m</sup> &#969;<sub>_i_</sub> &#8901; x<sub>_i_</sub>

Here, the Greek letter Sigma ∑ is used to represent **summation**. It simply means to evaluate the equation to the right 
multiple times and add up the results. In this case, the equation it will sum is w<sub>i</sub> &#8901; x<sub>i</sub>	 

But where do we get w<sub>i</sub> and x<sub>i</sub> ?

∑<sub>_i_=1</sub><sup>m</sup> means to iterate over all _i_ values, from 1 to m.

So to put it all together, &#8721;<sub>_i_=1</sub><sup>m</sup> &#969;<sub>_i_</sub> &#8901; x<sub>_i_</sub>  means the 
following:

* Start at _i_ = 1
* Evaluate &#969;<sub>_1_</sub> &#8901; x<sub>_1_</sub> and remember the results
* Move to _i_ = 2
* Evaluate &#969;<sub>_2_</sub> &#8901; x<sub>_2_</sub> and add these results to &#969;<sub>_1_</sub> &#8901; x<sub>_1_</sub>	 
* Continue repeating that process until _i_ = m, where _m_ is the number of inputs.

One last thing: you'll see equations written many different ways, both here and when reading on your own. For example, 
you will often just see ∑<sub>i</sub> instead of ∑<sub>_i_=1</sub><sup>m</sup>. The first is simply a shorter way of 
writing the second. That is, if you see a summation without a starting number or a defined end value, it just means 
perform the sum for all of the them. And sometimes, if the value to iterate over can be inferred, you'll see it as just 
∑. Just remember they're all the same thing: 

&#8721;<sub>_i_=1</sub><sup>m</sup> &#969;<sub>_i_</sub> &#8901; x<sub>_i_</sub> = &#8721;<sub>_i_</sub> &#969;<sub>_i_</sub> &#8901; x<sub>_i_</sub> = &#8721; &#969;<sub>_i_</sub> &#8901; x<sub>_i_</sub>

###### Calculating the Output with an Activation Function

Finally, the result of the perceptron's summation is turned into an output signal! This is done by feeding the linear 
combination into an **activation function**.

Activation functions are functions that decide, given the inputs into the node, what should be the node's output? 
Because it's the activation function that decides the actual output, we often refer to the outputs of a layer as its 
"activations".

One of the simplest activation functions is the **Heaviside step function**. This function returns a **0** if the linear 
combination is less than 0. It returns a **1** if the linear combination is positive or equal to zero. The [Heaviside 
step function](https://en.wikipedia.org/wiki/Heaviside_step_function) is shown below, where h is the calculated linear combination:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589cf7dd_heaviside-step-graph-2/heaviside-step-graph-2.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5895102f_heaviside-step-function-2/heaviside-step-function-2.gif)

In the university acceptance example above, we used the weights w<sub>grades</sub> = -1, w<sub>test</sub> = -0.2w. Since 
w<sub>grades</sub> and w<sub>test</sub> are negative values, the activation function will only return a 1 if grades and 
test are 0! This is because the range of values from the linear combination using these weights and inputs are \(−∞,0] 
(i.e. negative infinity to 0, including 0 itself).

It's easiest to see this with an example in two dimensions. In the following graph, imagine any points along the line or 
in the shaded area represent all the possible inputs to our node. Also imagine that the value along the y-axis is the 
result of performing the linear combination on these inputs and the appropriate weights. It's this result that gets 
passed to the activation function.

Now remember that the step activation function returns 11 for any inputs greater than or equal to zero. As you can see 
in the image, only one point has a y-value greater than or equal to zero – the point right at the origin, (0, 0):

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589d2f7e_example-before-bias/example-before-bias.png)

Now, we certainly want more than one possible grade/test combination to result in acceptance, so we need to adjust the 
results passed to our activation function so it activates – that is, returns 1 – for more inputs. Specifically, we need 
to find a way so all the scores we’d like to consider acceptable for admissions produce values greater than or equal to 
zero when linearly combined with the weights into our node.

One way to get our function to return 11 for more inputs is to add a value to the results of our linear combination, 
called a **bias**.

A bias, represented in equations as bb, lets us move values in one direction or another.

For example, the following diagram shows the previous hypothetical function with an added bias of +3. The blue shaded 
area shows all the values that now activate the function. But notice that these are produced with the same inputs as the 
values shown shaded in grey – just adjusted higher by adding the bias term:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589d3055_example-after-bias/example-after-bias.png)

Of course, with neural networks we won't know in advance what values to pick for biases. That’s ok, because just like 
the weights, the bias can also be updated and changed by the neural network during training. So after adding a bias, we 
now have a complete perceptron formula:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58951180_perceptron-equation-2/perceptron-equation-2.gif)

This formula returns 1 if the input (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>m</sub>) belongs to the accepted-to-university 
category or returns 0 if it doesn't. The input is made up of one or more [real numbers](https://en.wikipedia.org/wiki/Real_number), each one represented by 
x<sub>i</sub>, where _m_ is the number of inputs.

Then the neural network starts to learn! Initially, the weights ( &#969;<sub>i</sub>) and bias (b) are assigned a random 
value, and then they are updated using a learning algorithm like gradient descent. The weights and biases change so that 
the next training example is more accurately categorized, and patterns in data are "learned" by the neural network.

Now that you have a good understanding of perceptions, let's put that knowledge to use. In the next section, you'll 
create the AND perceptron from the _Neural Networks_ video by setting the values for weights and bias.

### 14. Why "Neuronal Network" ?

[![why nn](http://img.youtube.com/vi/zAkzOZntK6Y/0.jpg)](https://youtu.be/zAkzOZntK6Y "why nn")

### 15. Perceptrons as logical operators

###### Perceptrons as Logical Operators

In this lesson, we'll see one of the many great applications of perceptrons. As logical operators! You'll have the 
chance to create the perceptrons for the most common of these, the **AND**, **OR**, and **NOT** operators. And then, 
we'll see what to do about the elusive **XOR** operator. Let's dive in!

###### AND Perceptron

[![perceptrons as l o](http://img.youtube.com/vi/45K5N0P9wJk/0.jpg)](https://youtu.be/45K5N0P9wJk "perceptrons as l o")

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5912bf0e_and-quiz/and-quiz.png)

###### What are the weights and bias for the AND perceptron?

Set the weights (```weight1```, ```weight2```) and bias ```bias``` to the correct values that calculate AND operation as 
shown above.

````python
import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1.0
weight2 = 1.0
bias = -2.0


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_06.PNG)

###### OR Perceptron

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5912c102_or-quiz/or-quiz.png)

The OR perceptron is very similar to an AND perceptron. In the image below, the OR perceptron has the same line as the 
AND perceptron, except the line is shifted down. What can you do to the weights and/or bias to achieve this? Use the 
following AND perceptron to create an OR Perceptron.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5912c232_and-to-or/and-to-or.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_07.PNG)

###### NOT Perceptron

Unlike the other perceptrons we looked at, the NOT operation only cares about one input. The operation returns a ```0```
if the input is ```1``` and a ```1``` if it's a ```0```. The other inputs to the perceptron are ignored.

In this quiz, you'll set the weights (```weight1```, ```weight2```) and bias ```bias``` to the values that calculate the 
NOT operation on the second input and ignores the first input.

````python
import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -1.0
bias = 0.0


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_08.PNG)

[![XOR](http://img.youtube.com/vi/TF83GfjYLdw/0.jpg)](https://youtu.be/TF83GfjYLdw "XOR")

###### XOR Perceptron

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5912c2f1_xor/xor.png)

###### Quiz: Build an XOR Multi-Layer Perceptron

Now, let's build a multi-layer perceptron from the AND, NOT, and OR perceptrons to create XOR logic!

The neural network below contains 3 perceptrons, A, B, and C. The last one (AND) has been given for you. The input to 
the neural network is from the first node. The output comes out of the last node.

The multi-layer perceptron below calculates XOR. Each perceptron is a logic operation of AND, OR, and NOT. However, the 
perceptrons A, B, and C don't indicate their operation. In the following quiz, set the correct operations for the 
perceptrons to calculate XOR.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/59112a6b_xor-quiz/xor-quiz.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_09.PNG)

### 16. Perceptrons trick

In the last section you used your logic and your mathematical knowledge to create perceptrons for some of the most 
common logical operators. In real life, though, we can't be building these perceptrons ourselves. The idea is that we 
give them the result, and they build themselves. For this, here's a pretty neat trick that will help us.

[![perceptrons trick](http://img.youtube.com/vi/-zhTROHtscQ/0.jpg)](https://youtu.be/-zhTROHtscQ "perceptrons trick")

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/5912022e_perceptronquiz/perceptronquiz.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_10.PNG)

[![perceptrons trick 2](http://img.youtube.com/vi/fATmrG2hQzI/0.jpg)](https://youtu.be/fATmrG2hQzI "perceptrons trick 2")

###### Time for some math!

Now that we've learned that the points that are misclassified, want the line to move closer to them, let's do some math. 
The following video shows a mathematical trick that modifies the equation of the line, so that it comes closer to a 
particular point.

[![perceptrons trick 3](http://img.youtube.com/vi/lif_qPmXvWA/0.jpg)](https://youtu.be/lif_qPmXvWA "perceptrons trick 3")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/06_11.PNG)

(3 + 0.1 &#8901; &#946;) + (4 + 0.1 &#8901; &#946;) + (-10 + 0.1 &#8901; &#946;) = 0
&#8660; -3 + 0.3 &#8901; &#946; = 0
&#8660; 0.3 &#8901; &#946; = 3
&#8660; &#946; = 10

### 17. Perceptrons algorithms

And now, with the perceptron trick in our hands, we can fully develop the perceptron algorithm! The following video will 
show you the pseudocode, and in the quiz below, you'll have the chance to code it in Python.

[![perceptrons algorithm](http://img.youtube.com/vi/p8Q3yu9YqYk/0.jpg)](https://youtu.be/p8Q3yu9YqYk "perceptrons algorithm")

###### Coding the Perceptron Algorithm

Time to code! In this quiz, you'll have the chance to implement the perceptron algorithm to separate the following data 
(given in the file data.csv).

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/May/590d06dd_points/points.png)

Recall that the perceptron step works as follows. For a point with coordinates (p,q), label y&#770;, and prediction 
given by the equation y&#770; = step(&#969;<sub>1</sub> &#8901; x<sub>1</sub> + &#969;<sub>2</sub> &#8901; x<sub>2</sub> + b) 

* If the point is correctly classified, do nothing.
* If the point is classified positive, but it has a negative label, subtract &#945;p, &#945;q and &#945;, from &#969;<sub>1</sub>, &#969;<sub>2</sub>, and b respectively.
* If the point is classified negative, but it has a positive label, add &#945;p, &#945;q and &#945;, from &#969;<sub>1</sub>, &#969;<sub>2</sub>, and b respectively.

Then click on ```test run``` to graph the solution that the perceptron algorithm gives you. It'll actually draw a set of 
dotted lines, that show how the algorithm approaches to the best solution, given by the black solid line.

Feel free to play with the parameters of the algorithm (number of epochs, learning rate, and even the randomizing of the 
initial parameters) to see how your initial conditions can affect the solution!

### 18. Non-linear regions

[![NL regions](http://img.youtube.com/vi//0.jpg)]( "NL regions")

### 19. Error functions

[![err func](http://img.youtube.com/vi//0.jpg)]( "err func")

### 20. Log-loss error function

[![log loss err func](http://img.youtube.com/vi//0.jpg)]( "log loss err func")

### 21. Discrete vs Continuous

[![D vs C](http://img.youtube.com/vi//0.jpg)]( "D vs C")