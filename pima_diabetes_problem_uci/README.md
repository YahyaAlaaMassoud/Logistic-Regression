# **Pima Indians Diabetes Problem**
## Data Preprocessing
This dataset is composed of 768 rows, each rows represents a feature vector for a specific person, the feature are:
 - Pregnancies.
 - Glucose.
 - BloodPressure.
 - SkinThickness.
 - Insulin.
 - BMI.
 - DiabetesPedigreeFunction.
 - Age.
 - Whether a patient has diabetes or not.
 
To be able to read the dataset, I used **pandas** library.
```python
import pandas as pd
dataset = pd.read_csv('Pima Diabetes Dataset/diabetes.csv')
```
This is how the dataset is organized:<br/>
![dataset.](https://github.com/YahyaAlaaMassoud/Logistic-Regression/blob/master/pima_diabetes_problem_uci/images/dataset.png
"dataset")
<br/>
In order to get better accuracy with logistic regression, we have to choose only the features that will make a real impact on the
desicion of the classifier whether the person has diabetes or not, so in this case I've chosen all the columns as input features. 
And also extracted the last column as the output vector for the dataset.
```python
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 8].values
```
Then I had to split the dataset into training and test sets, with test set size equal to 25% of the total dataset size. To do so, 
I used **sklearn** library.
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
```
Now, to get better results with the classifier, it is always better to scale every feature so that each column has number in scale
from 0 to 1.
I used a **MinMaxScaler** from **sklearn** also, to tranform the features to scaled feature.<br/>
The transformation formula is *(from sklearn documentation)*
```python
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```
This is how I used it in the model
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Now all that is left in the preprocessing step is to assign the right shapes to the input and output matrices.
X_train's shape should be (8, 576).
X_test's shape should be (8, 192).
Y_train's shape should be (1, 576).
Y_test's shape should be (1, 192).

<hr/>

## Learning Process
I order to get a model that performs well on a specific dataset, you have to tune some hyperparameters to get the best working model, 
here I have tried to learn features with 4 learning rates, and compared the result of training every model for 2500 epochs.
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'logistic_regression_class')))
from logistic_regression_class.logistic_regression_class import LogisticRegression

learning_rates = [1, 0.5, 0.05, 0.005]
costs = []
confusion_matrix_train = []
confusion_matrix_test = []
for i in learning_rates: 
    classifier = LogisticRegression(X_train, Y_train, i, 2500, True) 
    costs.append((i, classifier.optimize()))
    Y_prediction_train = classifier.predict(X_train, Y_train)
    confusion_matrix_train.append(classifier.get_confusion_matrix())
    Y_prediction_test = classifier.predict(X_test, Y_test)
    confusion_matrix_test.append(classifier.get_confusion_matrix())
```
>cost at iteration 0 : 0.6691 accuracy: 64.2361%<br/>
>cost at iteration 100 : 0.5653 accuracy: 71.0069%<br/>
>.<br/>
>.<br/>
>cost at iteration 2500 : 0.4826 accuracy: 76.3889%<br/>
>![plot](https://github.com/YahyaAlaaMassoud/Logistic-Regression/blob/master/pima_diabetes_problem_uci/images/train_accuracy.png
"plot")<br/>
>***This is an example of running the classifier on the training set with learning rate equal to 1.***


### Comparing classifiers
In order to compare the classifier with different learning rates, we have to plot the learning graph for each of the classifiers.<br/>
 - The **cost** array have 4 tuples, each contains all costs for classifier with learning rate *i*, and the leaning rate *i*. 
 (costs, learning_rate).
```python
import matplotlib.pyplot as plt

for i in costs:
    learning_rate, cost = i
    plt.plot(np.squeeze(cost), label= "learning rate = " + str(learning_rate))
plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```
Now we get a graph that comapres the 4 classifier with 4 different learning rates.<br/>
![compare](https://github.com/YahyaAlaaMassoud/Logistic-Regression/blob/master/pima_diabetes_problem_uci/images/comparing_classifiers.png
"compare")<br/>
After comparing the classifiers, it is obvious now to choose a learning rate either ***1 or 0.5***.<br/>
Now let's calculate the train and test accuracies for each of the 4 classifiers using the two arrays, **confustion_matrix_train** 
and **confustion_matrix_test**.
```python
classifier_number = 0
for confusion_mat in confusion_matrix_train:
    train_accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    print("train accuracy in classifier " + str(classifier_number) + ": " + str(np.round(train_accuracy * 100, 4)) + '%')
    classifier_number += 1
```
Will result in:
>train accuracy on classifier 0: 76.3889%<br/>
>train accuracy on classifier 1: 76.0417%<br/>
>train accuracy on classifier 2: 71.875%<br/>
>train accuracy on classifier 3: 64.2361%<br/>
```python
classifier_number = 0
for confusion_mat in confusion_matrix_test:
    test_accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    print("test accuracy on classifier " + str(classifier_number) + ": " + str(np.round(test_accuracy * 100, 4)) + '%')
    classifier_number += 1    
```
Will result in:
>test accuracy on classifier 0: 80.2083%<br/>
>test accuracy on classifier 1: 79.1667%<br/>
>test accuracy on classifier 2: 75.0%<br/>
>test accuracy on classifier 3: 67.7083%<br/>

**If you have any question don't hesitate to contact me on my email: *yahyaalaamassoud@gmail.com*.**
**Thank you.**
<hr/>

