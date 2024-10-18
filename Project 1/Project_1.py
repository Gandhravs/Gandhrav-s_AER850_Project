# -*- coding: utf-8 -*-
"""
Project 1
Created on Wed Oct  9 11:11:28 2024

@author: gshar """

# 2.1 Step 1: Data Processing - 2 Marks
# Read data from a csv file and convert that into a dataframe, which will allow for all further
# analysis and data manipulation. # Package(s) required: Pandas

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
import sklearn as skl
from sklearn.model_selection import train_test_split


"""Step 1 Data-Processing"""

df = pd.read_csv(r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav-s_AER850_Project\Project 1\Project_1_Data.csv");

"""Step 2 Raw Data-visualization"""

# Creating the figure 
fig = plt.figure();
ax = fig.add_subplot(111,projection='3d'); # 1 row 1 column 1 subplot
print(type(ax));# describing 3D subplot

ax.scatter3D(df['X'],df['Y'],df['Z']);
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('z');
# ax.view_init(35,185); # (elevation angle, azimuth angle in deg)
two_d = df.hist();
print(df.describe());

# 2.3 Step 3: Correlation Analysis - 15 Marks
# Assess the correlation of the features with the target variable.

"""Step 3 Correlation Analysis"""
correlation_matrix = df.corr(); # calculating the correlation variables 
plt.figure(figsize=(15,8));
print(correlation_matrix);
sns.heatmap(correlation_matrix,cmap='BuPu',vmin = -1, vmax =1,annot=True,square = True, annot_kws={'fontsize':11, 'fontweight':'bold'}); # passed a dictionary for annot_kws
plt.title("Pearson Correlation Matrix");


# 2.4 Step 4: Classification Model Development/Engineering - 20 Marks

"""Step 4 Classification Model"""

# first I need to identify my x and y variables

coord =df[['X','Y','Z']];       
target = df['Step'];

X_train, X_test, y_train, y_test = train_test_split(coord,target,random_state = 42, test_size = 0.2);
print(X_train.shape)
print(y_train.value_counts(normalize=True)*100)

print(y_test.value_counts(normalize=True)*100)
# now the data has been split I can begin the three different models. 



# 3 Logistic Regression

logistic_model = LogisticRegression(max_iter=1000) # to allow it to converge properly
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)
logistic_model.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # use model to make prediction
# y_train_pred_svm = best_model_svm.predict(X_train)


# y_test_pred_svm = best_model_svm.predict(X_test)

# # Debugging the predictions
# print("y_test:", y_test)
# print("y_test_pred_svm:", y_test_pred_svm)

# # Use model to make prediction and print accuracy
# print("Train Accuracy:", accuracy_score(y_train, y_train_pred_svm))
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))









