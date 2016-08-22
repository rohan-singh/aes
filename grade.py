import pandas as pd
import numpy as np
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.naive_bayes import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

train_data_df = pd.read_csv('features.csv',delimiter=',',header = 0)

X_train, X_test, y_train, y_test  = train_test_split(train_data_df, train_data_df.grade, random_state= 90 ,train_size = 0.75)

model = LinearSVC()
model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = MultinomialNB(alpha = 0.8)
model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = DecisionTreeClassifier(max_depth=4, min_samples_split=4, min_samples_leaf=4)
model = model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
