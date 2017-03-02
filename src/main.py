import pandas as pd
from prepare import *

from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

#####################################
# Fit the model
#####################################

# Load data
X, y = load_data()

# Subset Train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
		X, y, test_size=0.2, random_state=0)

# Data format: from pandas object to numpy array
X_train = X_train.values
y_train = y_train.values.ravel()

# First model: Logistic regression
logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
score1 = logistic.score(X_test, y_test)

# Second model: Random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
score2 = rf.score(X_test, y_test)

# Comparing scores
print('Logistic Regression:')
print(score1)
print('Random Forest:')
print(score2)

# It is possible to chain many others classifer algorithm and pick the best one
# for your future predictions. See: http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
