# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:23:16 2024

@author: gshar
"""

# 2.1 Step 1: Data Processing - 2 Marks
# Read data from a csv file and convert that into a dataframe, which will allow for all further
# analysis and data manipulation. # Package(s) required: Pandas

# importing all the libraries for the project
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score,precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import joblib

"""Step 2.1 Data-Processing"""

# reading the file as a csv file and converting to a data frame
df = pd.read_csv(r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav-s_AER850_Project\Project 1\Project_1_Data.csv");
print(df.head())


"""Step 2.2 Raw Data-visualization"""

# Creating the figure 
fig = plt.figure();
ax = fig.add_subplot(111,projection='3d'); # 1 row 1 column 1 subplot
print(type(ax));# describing 3D subplot 

# setting the 3D scatter plot's labels

ax.scatter3D(df['X'],df['Y'],df['Z']);
ax.set_xlabel('x'); # x label
ax.set_ylabel('y'); # y label
ax.set_zlabel('z'); # z label
ax.view_init(35,185); # (elevation angle, azimuth angle in deg)

#extra histogram
two_d = df.hist();
print(df.describe()); # summary of statistics presents the mean, std dev etc of the model

# 2.3 Step 3: Correlation Analysis - 15 Marks
# Assess the correlation of the features with the target variable.

"""Step 2.3 Correlation Analysis"""
correlation_matrix = df.corr(); # calculating the correlation variables 
plt.figure(figsize=(15,8));  # creates a larger size for the heatmap
print(correlation_matrix);
sns.heatmap(correlation_matrix,cmap='BuPu',vmin = -1, vmax =1,annot=True,square = True, annot_kws={'fontsize':11, 'fontweight':'bold'}); # passed a dictionary for annot_kws
plt.title("Pearson Correlation Matrix");



# 2.4 Step 4: Classification Model Development/Engineering - 20 Marks

"""Step 2. 4 Classification Model"""

# first I need to identify my x and y variables

coord =df[['X','Y','Z']] # these are the features
target = df['Step']; # these are the target

# splitting the data set into train and test using the 80-20 split convention and stratifying the samples
X_train, X_test, y_train, y_test = train_test_split(coord,target,random_state = 42, test_size = 0.2,stratify=target);



# 1 Random Forest hyperParameters

rf = RandomForestClassifier(random_state=42) 

# Define a grid of hyperparameters for tuning the Random Forest model
param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform a grid search with k-fold cross-validation 
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=10, scoring='accuracy', n_jobs=-1)  # n_jobs mean all cpus on pc are used
grid_search_rf.fit(X_train, y_train)  # Fit the model on training data
best_model_rf = grid_search_rf.best_estimator_   # Get the best model from grid search
print("\n Best Random Forest parameters are:", best_model_rf)




# 2 SVM Model

svm_model = SVC();  # initializing model

param_grid_svm = [
    {'kernel':['rbf','sigmoid','linear'] ,'C':[0.1,1,10,100,1000],'gamma':['scale','auto']}
]

grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train,y_train)
best_model_svm = grid_search_svm.best_estimator_
print('\n the best parameters are:',grid_search_svm.best_params_)




# 3 Logistic Regression

logistic_model = LogisticRegression(max_iter=1000) # to allow it to converge properly
logistic_model.fit(X_train,y_train)
y_pred_Lr = logistic_model.predict(X_test)
logistic_model.score(X_test, y_test)



# Decision Tree with Randomized 

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search_dt = RandomizedSearchCV(decision_tree, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1,random_state = 42)
random_search_dt.fit(X_train, y_train)
best_model_dt = random_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)



"""Step 2.5 Model Performance Analysis"""



# Random Forest 


# use model to make prediction
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)




# 3. Classification Report for Random Forest
class_report_rf = classification_report(y_test, y_test_pred_rf)
print("Classification Report for Random Forest:\n", class_report_rf)

"""Checking for Overfitting"""
train_accuracy = accuracy_score(y_train,y_train_pred_rf);
test_accuracy = accuracy_score(y_test,y_test_pred_rf);




#######
# Classification Report for SVM


y_train_pred_svm = best_model_svm.predict(X_train);
y_test_pred_svm = best_model_svm.predict(X_test);

class_report_svm = classification_report(y_test, y_test_pred_svm)
print("\n\n Classification Report for SVM: \n", class_report_svm)

"""Checking for Overfitting"""
train_accuracy_svm = accuracy_score(y_train,y_train_pred_svm);
test_accuracy_svm = accuracy_score(y_test,y_test_pred_svm);



####
# Classification Report for LR

y_train_pred_Lr = logistic_model.predict(X_train);
y_test_pred_Lr = logistic_model.predict(X_test);


class_report_Lr_test = classification_report(y_test, y_test_pred_Lr)
print("\n\n Classification Report for Lr (Test): \n", class_report_Lr_test)

"""Checking for Overfitting"""
train_accuracy_Lr = accuracy_score(y_train,y_train_pred_Lr);
test_accuracy_Lr = accuracy_score(y_test,y_test_pred_Lr);


##### 
# Classification Report for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train);
y_test_pred_dt= best_model_dt.predict(X_test);

class_report_dt = classification_report(y_test, y_test_pred_dt)
print("\n\n Classification Report for DT: \n", class_report_dt)

"""Checking for Overfitting"""
train_accuracy_dt = accuracy_score(y_train,y_train_pred_dt);
test_accuracy_dt = accuracy_score(y_test,y_test_pred_dt);


# Selected model is Decision tree

# Confusion matrix

conf_matrix_dt = confusion_matrix(y_test, y_test_pred_dt)
class_labels = best_model_dt.classes_  # extracting the labels from the trained model to label 
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt,display_labels=class_labels)
disp.plot(cmap='viridis')

plt.show()  # show the confusion matrix


""" Step 2.6 Model Stacking  """


# Define estimators for stacking
estimators = [
    ('lr', logistic_model), 
    ('Rf',best_model_rf),]

# Final estimator (SVM) for stacking
final_estimator = best_model_svm
sc = StackingClassifier(estimators = estimators, 
                        final_estimator= final_estimator)

# Train stacking model
sc.fit(X_train,y_train)

# predict model

y_sc_pred = sc.predict(X_test);

# evaluating performance
accuracy = accuracy_score(y_test, y_sc_pred)
precision = precision_score(y_test, y_sc_pred, average = 'weighted')
f1 = f1_score(y_test, y_sc_pred, average = 'weighted')

# Output stacking model metrics
print(f"stacking model accuracy: {accuracy}")
print(f"stacking model Precision: {precision}")
print(f"stacking model F1 Score: {f1}")

stacked_conf_matrix = confusion_matrix(y_test, y_sc_pred)
print("Confusion Matrix for Stacked Model:\n", stacked_conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=stacked_conf_matrix, display_labels=class_labels)
disp.plot(cmap='plasma')
plt.show()



""" Step 2.7 Model Evaluation"""

# Save and Load the SVM model using joblib

joblib.dump(best_model_svm,'svm_job.joblib') # save the model under joblib format in new file called svm_job
loaded_svm_job=joblib.load('svm_job.joblib') # loading it to return your model object used to make predictions

# prediction coordinates
coords_to_predict = np.array([[9.375, 3.0625, 1.51],
[6.995, 5.125, 0.3875],
[0, 3.0625, 1.93],
[9.4, 3, 1.8],
[9.4, 3, 1.3]])

coords_to_predict_df = pd.DataFrame(coords_to_predict,columns =['X','Y','Z']) # converting numpy to pandas df

mj = loaded_svm_job.predict(coords_to_predict_df)  # Predictions for new data

print(mj)


"""Checking overfitting with print statements """
# Decision Tree Accuracy
print(f"Decision Tree Training Accuracy: {train_accuracy_dt:.4f}")
print(f"Decision Tree Test Accuracy: {test_accuracy_dt:.4f}")

# Logistic Regression Accuracy
print(f"Logistic Regression Training Accuracy: {train_accuracy_Lr:.4f}")
print(f"Logistic Regression Test Accuracy: {test_accuracy_Lr:.4f}")

# SVM Accuracy
print(f"SVM Training Accuracy: {train_accuracy_svm:.4f}")
print(f"SVM Test Accuracy: {test_accuracy_svm:.4f}")

# Random Forest Accuracy
print(f"Random Forest Training Accuracy: {train_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")



