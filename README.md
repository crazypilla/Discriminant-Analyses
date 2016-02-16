Problem 1 (10 code + 10 report = 20 points) Experiment with
Gaussian discriminators
Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). Refer
to Lecture 20 slides and handouts. Implement two functions in Python: ldaLearn and qdaLearn which take
a training data set (a feature matrix and labels) and return the means and covariance matrix (or matrices).
Implement two functions ldaTest and qdaTest which return the true labels for a given test data set and
the accuracy using the true labels for the test data. The format of arguments and the outputs is provided
in the base code.

Problem 2 (5 code + 5 report = 10 Points): Experiment with
Linear Regression
Implement ordinary least squares method to estimate regression parameters by minimizing the squared loss.

Note that this is same as maximizing the log-likelihood in the Bayesian setting. You need to implement the
function learnOLERegression. Also implement the function testOLERegression to apply the learnt weights
for prediction on both training and testing data and to calculate the root mean squared error (RMSE):

Problem 3 (10 code + 10 report = 20 Points): Experiment with Ridge Regression
You need to implement it in the function learnRidgeRegression.

Problem 4 (20 code + 5 report = 25 Points): Using Gradient Descent
for Ridge Regression Learning
As discussed in class, regression parameters can be calculated directly using analytical expressions (as in
Problem 2 and 3). In this problem, you have to implement the
gradient descent procedure for estimating the weights w.
You need to use the minimize function (from the scipy library) which is same as the minimizer that you
used for rst assignment. You need to implement a function regressionObjVal to compute the regularized
squared error (See (3)) and its gradient with respect to w. In the main script, this objective function will
be used within the minimizer.

Problem 5 (10 code + 5 report = 15 Points): Non-linear Regression
In this problem we will investigate the impact of using higher order polynomials for the input features. For
this problem use the third variable as the only input variable:
x t r a i n = x t r a i n [ : , 3 ]
x t e s t = x t e s t [ : , 3 ]
Implement the function mapNonLinear.m which converts a single attribute x into a vector of p attributes,


Problem 6 (0 code + 10 report = 10 points) Interpreting Results
Using the results obtained for previous 4 problems, make nal recommendations for anyone using regression
for predicting diabetes level using the input features.
