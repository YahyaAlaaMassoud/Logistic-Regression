# **Logistic-Regression**
This repo contains my own implementation for ***Logistic Regression***, and examples on applying it to different datasets with 
explanation for each example of the data preprocessing step, and the learning algorithm behavior.<br/>
### What is Logistic Regression?
**Logistic Regression** is a supervised learning technique that is used for binary classification problems, where the dataset 
conatins one or more independant varibales that determine a binary outcome (0 or 1).<br/>
In a logistic regression classifier, you may want to input a feature vector X which describes the features for a single row of data,
and you want to predict a binary output value which is either 0 or 1.<br/>
More formally, given an input vector X, you want to predict y_hat which is an output vector describing the probability that y = 1 given
feature vector X, **y_hat = p(y = 1 / X)**.
##### For example:
> You have an input vector X, where the features are gender, age and salary for a specific person, and you want to predict whether 
or not this person will purchase a specific product or not.

<hr/>

### How data is prepared to be fed into the classifier?
The training dataset will contain rows of data, where each row represents a tuple of (X, y), where:
 - **X** is a n_x dimentional vector (n_x is the number of independant varibales discribing the row of data).
 - **y** is a binary value, whether 0 or 1, describing whether or not the user has purchased the product.
 
In order to train the **Logistic Regression Classifier**, we'll divide our dataset into **training and test sets**, having **m** training examples.
We'll then stack every training example **X(i)** as column vectors in a large input matrix of shape **(n_x, m)**, and also stack the output 
values **y** as columns in a large output matrix of shape **(1, m)**.<br/>
We'll have also to initalize a **weights vector** and a **bias** which are learnable, and both will allow the classifier to learn and extract 
features and paterns from the input data.<br/>
Then all what is left to do is to feed this data into our Logistic Regression Classifier, the image below describes how to the data is fed
to the classifier.<br/>
![Logistic Regression Classifier.](https://github.com/YahyaAlaaMassoud/Logistic-Regression/blob/master/images/perceptron_node.png
"Summarization Result")

### What is the net input function?
Taking the dot product of a given feature vector and the vector of weights, will result in a single value output that describes the 
contribution of the initialized weights in the result of the classifier. But this output value does not represent any expected value, neither
0 or 1, that's why we have to pass this value into another function that will map this value to another value between 0 and 1.<br/>
Here comes the power of the **activation function**.
### What is the activation function?


