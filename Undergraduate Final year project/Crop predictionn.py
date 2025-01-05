#!/usr/bin/env python
# coding: utf-8

# # Crop prediction

# ###  Importing necessary libraries 

# In[1]:


import pandas as pd
import numpy as np
import numpy
import math 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# ### Importing dataset

# In[2]:


dataset= pd.read_csv('dataset1.csv') 


# ### Displaying the shape of the dataset

# In[3]:


print(dataset.shape)


# ### We have four different crops in our data set, but in order to achieve a better classification accuracy, we'll be dropping crop 3 and 4 which contains only 19.6% of the total records making them irrelevant. So we will be focussing solely on the binary classification between  crop 1 and 2. i.e. yam and cassava.

# In[4]:


# Filter the dataset for only yam (1) and cassava (2)
relevant_labels = [1, 2]
new_data = dataset[dataset['CROP'].isin(relevant_labels)]

# Reset the index of the filtered dataset
new_data.reset_index(drop=True, inplace=True)

print("Now we can focus the classification task on", relevant_labels)
print("Our current data now has", new_data.shape, "rows and columns")


# ### Displaying the first five rows of the dataset

# In[5]:


#displaying first 5  rows of the dataset
new_data.head()


# ### Summary statistics

# In[6]:


#dataset description
new_data.describe()


# Based on the output:
# MIN_TEMP:
# 
# count: There are 346 non-null values in this column.
# mean: The average minimum temperature is approximately 31.21.
# std: The standard deviation is approximately 2.93, indicating the variability in the minimum temperature values.
# min: The minimum observed minimum temperature is 20.30.
# 25%: 25% of the data has minimum temperatures below 29.23.
# 50%: The median (middle value) of the minimum temperature values is 31.30.
# 75%: 75% of the data has minimum temperatures below 33.30.
# max: The maximum observed minimum temperature is 38.00.
# MAX_TEMP:
# 
# Similar to MIN_TEMP, these statistics represent the maximum temperature values.
# RH150:
# 
# These statistics represent the relative humidity at a height of 150cm.
# The average relative humidity is approximately 87.15.
# The minimum, 25%, 50%, and 75% values indicate the distribution of relative humidity values.
# The maximum relative humidity observed is 97.00.
# RH60:
# 
# These statistics represent the relative humidity at a height of 60cm.
# The average relative humidity is approximately 102.06.
# The minimum, 25%, 50%, and 75% values indicate the distribution of relative humidity values.
# The maximum relative humidity observed is 344.10.
# Rainfall:
# 
# These statistics represent the amount of rainfall.
# The average rainfall is approximately 9.17.
# The minimum, 25%, 50%, and 75% values indicate the distribution of rainfall amounts.
# The maximum observed rainfall is 20.40.
# Area cropped (Ha):
# 
# These statistics represent the area of the crop cultivated in hectares.
# The average area cropped is approximately 0.13 Ha.
# The minimum, 25%, 50%, and 75% values indicate the distribution of area cropped.
# The maximum observed area cropped is 0.36 Ha.
# Yield (Mt/Ha):
# 
# These statistics represent the yield of the crop in metric tons per hectare.
# The average yield is approximately 0.76 Mt/Ha.
# The minimum, 25%, 50%, and 75% values indicate the distribution of yield values.
# The maximum observed yield is 0.99 Mt/Ha.
# MIN_NDVI:
# 
# These statistics represent the minimum Normalized Difference Vegetation Index (NDVI) values.
# The average minimum NDVI is approximately 1.51.
# The minimum, 25%, 50%, and 75% values indicate the distribution of NDVI values.
# The maximum observed minimum NDVI is 2.00.
# MAX_NDVI:
# 
# Similar to MIN_NDVI, these statistics represent the maximum NDVI values.
# CROP:
# 
# These statistics represent the crop labels.
# The column is not numerical, so descriptive statistics like mean and standard deviation are not applicable.
# The minimum and maximum values indicate the range of crop labels observed (1 for yam, 2 for cassava).
# These statistics provide a summary of the distribution and characteristics of each column in the dataset. They help in understanding the central tendency, variability, and range of the numerical features.
# 
# 
# 
# 
# 
# 

# ##### Transposed summary statistics table for the dataset

# In[7]:


new_data.describe().transpose()


# #### Let's check for missing values

# In[8]:


# Check for missing values
missing_values = new_data.isnull()

# Count the number of missing values in each column
missing_counts = missing_values.sum()

# Print the columns with missing values and their respective counts
print(missing_counts[missing_counts > 0])


# No missing values detected

# #### Checking if there is any categorical data in the dataset or if there are numbers represented as strings, we can iterate over the columns of the dataset and check the data type of each column. If a column contains string values, it can be considered as potentially categorical or containing numbers represented as strings.

# In[9]:


# Iterate over each column in the dataset
for column in new_data.columns:
    # Check if the column contains string values
    if new_data[column].dtype == 'object':
        print(f"{column}: Categorical data or numbers represented as strings")
    else:
        print(f"{column}: Numerical data")


# Based on the output, it seems that the columns "MIN_TEMP," "Area cropped(Ha)," and "CROP" may contain categorical data or numbers represented as strings. The rest of the columns are identified as numerical data.

# #### Check unique values: For the columns identified as potential categorical data or numbers represented as strings, you can check the unique values to gain more insight into the data. This can help determine if the values are indeed categorical or if there are inconsistencies in how the numbers are represented.

# In[10]:


print(new_data["MIN_TEMP"].unique())
print(new_data["Area cropped(Ha)"].unique())
print(new_data["CROP"].unique())


# 

# **Let's convert the data types: If you determine that the columns should be treated as numerical data, you can convert them from strings to numeric values using the pd.to_numeric() function**

# In[11]:


new_data["MIN_TEMP"] = pd.to_numeric(new_data["MIN_TEMP"], errors="coerce")
new_data["Area cropped(Ha)"] = pd.to_numeric(new_data["Area cropped(Ha)"], errors="coerce")
new_data["CROP"] = pd.to_numeric(new_data["CROP"], errors="coerce")


# #### Now that the convertion is done, let's check again to see if any of the features still remains categorical or a string

# In[12]:


# Iterate over each column in the dataset
for column in new_data.columns:
    # Check if the column contains string values
    if new_data[column].dtype == 'object':
        print(f"{column}: Categorical data or numbers represented as strings")
    else:
        print(f"{column}: Numerical data")


# The output shows that all the numbers represented as strings has been converted to numerical

# #### Now, let's check for missing values, and if some exist, let's the rows 

# In[13]:


# Check for missing values
missing_values = new_data.isnull()

# Count the number of missing values in each column
missing_counts = missing_values.sum()

# Print the columns with missing values and their respective counts
print(missing_counts[missing_counts > 0])


# The output shows that, the are 3 missing records in the min temperature column and 1 missing record in the area cropped

# #### Let's the rows with the missing value and display the current shape of the dataset

# In[14]:


new_data.dropna(inplace=True)
print(new_data.shape)


# After dropping the rows with the missing values, the dataset now has 654 rows and 10 columms. 

# ## Visualize the distribution of the target variable using a bar plot or count plot

# ### Identify our classes in the target variable

# In[15]:


crop_values = new_data['CROP'].unique()
print(crop_values)


# The above output shows that our target variable has two classes crop 1 which is cassava and crop 2 which is yam

# ### Counting the total number of records for each class  in the target variable

# In[16]:


crop_counts = new_data['CROP'].value_counts()
print(crop_counts)


# The above output shows that crop 2(yam) has 346 records, and crop 1(cassava) has 308 records

# ### Plot the crop distribution using a bar chart

# In[17]:


# Calculate the count of each crop category
crop_counts = new_data['CROP'].value_counts()

# Plot the distribution using a bar chart
sns.barplot(x=crop_counts.index, y=crop_counts.values, palette=['orange', 'blue'])
plt.xlabel('Crop')
plt.ylabel('Count')
plt.title('Distribution of Crops')
plt.xticks(ticks=[0, 1], labels=['Yam', 'Cassava'])
plt.show()


# #### Correlation between the variables in the dataset

# In[18]:


corr=new_data.corr()
print(corr)


# Based on the output:
# 
# MIN_TEMP and MAX_TEMP: These variables have a negative correlation coefficient of -0.36, indicating a weak negative correlation. As MIN_TEMP increases, MAX_TEMP tends to decrease, and vice versa.
# 
# MIN_TEMP and RH150: These variables have a weak negative correlation with a coefficient of -0.08. As MIN_TEMP increases, RH150 tends to decrease slightly.
# 
# MAX_TEMP and RH150: These variables have a stronger negative correlation with a coefficient of -0.63. As MAX_TEMP increases, RH150 tends to decrease significantly.
# 
# RH150 and RH60: These variables have a strong positive correlation coefficient of 0.78. As RH150 increases, RH60 also tends to increase.
# 
# RH150 and Rainfall: These variables have a positive correlation coefficient of 0.61, indicating a moderate positive correlation. As RH150 increases, Rainfall tends to increase.
# 
# Area cropped(Ha) and Yield (Mt/Ha): These variables have a positive correlation coefficient of 0.51, indicating a moderate positive correlation. As the area cropped increases, the yield tends to increase.
# 
# MIN_NDVI and MAX_NDVI: These variables have a negative correlation coefficient of -0.28, indicating a weak negative correlation. As MIN_NDVI increases, MAX_NDVI tends to decrease.
# 
# CROP and MIN_TEMP: These variables have a weak positive correlation coefficient of 0.14. As MIN_TEMP increases, CROP tends to increase slightly.

# 

# #### Heatmap to visualize the correlation matrix between the variables in the dataset

# In[19]:


fig = plt.figure(figsize = (12,9))
sns.heatmap(corr,annot=True,cmap='RdYlGn')


# # Feature engineering

# ### Separate the features (x) and target variable (y) 

# In[20]:


x = new_data.drop('CROP', axis=1)  
y = new_data['CROP']


# #### split the dataset into training (70%) and testing (30%) sets

# In[21]:


X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)


# #### printing the shape of training and testing dataset

# In[22]:


print('X train =',X_train.shape,' Y train =',Y_train.shape )
print('x test =',x_test.shape,' y test =',y_test.shape)


# # k nearest neighbor classifier	
# ### training the model 

# In[23]:


knn_model = KNeighborsClassifier()

# Train the model using the training data
knn_model.fit(X_train, Y_train)

# Make predictions on the training data
y_pred_training = knn_model.predict(X_train)


# ### training reoprt

# In[24]:


(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# Precision: A higher precision value suggests a lower rate of false positives. In this case, the precision values for both classes are around 0.74-0.75, which can be considered reasonably good.
# 
# Recall:  A higher recall value indicates a lower rate of false negatives. In this case, the recall values for both classes are around 0.73-0.75, which can be considered reasonably good.
# 
# F1-score: A higher F1-score indicates a better balance between precision and recall. In this case, the F1-scores for both classes are 0.74, indicating a reasonable balance between precision and recall.
# 
# Accuracy: An accuracy value of 0.74 suggests that the model predicts the correct class for 74% of the instances.
# 
# In summary, based on the output, the model's performance can be considered reasonably good.

# ### testing  the model 

# In[25]:


y_pred1 = knn_model.predict(x_test)


# ### testing report 

# In[26]:


(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))                      


# Precision:the precision for class 1 is 0.74, meaning that out of all instances predicted as class 1, 74% were correct. For class 2, the precision is 0.74, indicating that 74% of instances predicted as class 2 were correct.
# 
# Recall: the recall for class 1 is 0.66, meaning that the model identified 66% of the true instances of class 1. For class 2, the recall is 0.81, indicating that the model identified 81% of the true instances of class 2.
# 
# F1-score: the F1-score for class 1 is 0.70, representing an overall measure of accuracy for class 1. For class 2, the F1-score is 0.77, indicating the overall accuracy for class 2.
# 
# Support: Support represents the number of instances in each class in the dataset. In this case, the support for class 1 is 89, indicating the number of instances in class 1. For class 2, the support is 108, representing the number of instances in class 2.
# 
# Accuracy: the accuracy is 0.74, meaning the model predicted the correct class for 74% of the instances.
# 
# Macro average:the macro average precision, recall, and F1-score are all around 0.74, indicating the average performance across both classes.
# 
# Weighted average: the weighted average precision, recall, and F1-score are all around 0.74, taking into account the support for each class.
# 
# Overall, the precision, recall, and F1-scores for both classes are relatively moderate, and the accuracy is around 0.74. This suggests that the model's performance is modest, and it may have some difficulty accurately predicting instances in both classes.

# ### Testing the model with a set of records that does not exist in the original dataset

# In[27]:


# New record to be classified
new_record = [[22.7, 31.5, 59, 65, 42.5, 3412, 8.3, 0.1455, 0.9854]]  # Example features for a new record

# Make predictions on the new record using the trained classifier
y_pred_new = knn_model.predict(new_record)

print("Predicted class label for the new record: {}".format(y_pred_new))


# When the model was tested with a set of records( 22.7, 31.5, 59, 65, 42.5, 3412, 8.3, 0.1455, 0.9854) oustide the original dataset predicted crop 1 to should be grown 

# ### cross validation test for knn

# In[28]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result1 = model_selection.cross_val_score(knn_model, X_train, Y_train, cv=kfold)


# ### testing for accuracy
# 

# In[29]:


result1.mean()*100


# Base on the output, the mean cross-validation accuracy is approximately 64.2%, indicating that the KNN model achieved an average accuracy of 52.31% on the validation sets during the cross-validation process.

# ### generating predictions with the model using the value of x test
# 
#   A confusion matrix is a table that summarizes the performance of a classification model by displaying the counts of true positive, true negative, false positive, and false negative predictions.
# 

# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat1=confusion_matrix(y_test,y_pred1)
conf_mat1


# The value 59 represents the count of true negatives (TN). These are instances that were correctly predicted as the negative class.
# 
# The value 30 represents the count of false positives (FP). These are instances that were incorrectly predicted as the positive class.
# 
# The value 21 represents the count of false negatives (FN). These are instances that were incorrectly predicted as the negative class.
# 
# The value 87 represents the count of true positives (TP). These are instances that were correctly predicted as the positive class.
# 
# Here is a breakdown of the matrix
# 
#                     Predicted Negative   Predicted Positive
#                     
#      Actual Negative           59 (TN)             30 (FP)
# 
#      Actual Positive           21 (FN)             87 (TP)
# 

# ### Visualizing confusion_matrix for training

# In[31]:


ax = plt.subplot()
sns.heatmap(conf_mat1, annot=True, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('Actual Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['cassava','yam'])
ax.yaxis.set_ticklabels(['yam','cassava'])



# ### Printing the Evaluation Metrics of the Model
# 

# In[32]:


from sklearn.metrics import r2_score

MSE1 = np.square(np.subtract(y_test, y_pred1)).mean()
RMSE1 = math.sqrt(MSE1)
R_squared1 = r2_score(y_test, y_pred1)
print("Mean Square Error: ", MSE1)
print("Root Mean Square Error: ", RMSE1)
print("Coefficient of Determination: ", R_squared1)


# Mean Square Error (MSE): A lower MSE indicates better model performance, as it signifies smaller errors between the predicted and actual values. In this case, the MSE is 0.25888324873096447.
# 
# Root Mean Square Error (RMSE):  Like MSE, a lower RMSE indicates better model performance. In this case, the RMSE is 0.5088057082334715, representing the average magnitude of the errors between the predicted and actual values.
# 
# Coefficient of Determination (R^2): In this case, the R^2 is -0.045255930087390706, suggesting that the model performs worse than a horizontal line and has a poor fit to the data.

# # DECISION TREE
# 

# In[33]:


dtc= DecisionTreeClassifier(criterion = "entropy", min_samples_leaf=2,
                            random_state = 100)
dtc_model = dtc.fit(X_train,Y_train)


# ### TRAINING MODEL

# In[34]:


y_pred_training = dtc_model.predict(X_train)


# 

# ### training report

# In[35]:


(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# 

# ### Testing model 

# In[36]:


y_pred2 = dtc_model.predict(x_test)


# In[37]:


# New record to be classified
new_record1 = [[22.2, 33, 54, 90, 12.3, 3412, 1.3, 0.1955, 0.8854]]  # Example features for a new record

# Make predictions on the new record using the trained classifier
y_pred_new = dtc_model.predict(new_record)

print("Predicted class label for the new record: {}".format(y_pred_new))


# ### testing report 

# In[38]:


(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# 

# ### cross validation 

# In[39]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result2 = model_selection.cross_val_score(dtc, X_train, Y_train, cv=kfold)
result2


# These accuracy scores indicate the performance of the model on different subsets of the data. It appears that the model achieved varying levels of accuracy across the folds, ranging from 0.375 to 0.875. The average accuracy can be calculated by taking the mean of these scores, which would give an overall estimate of the model's performance.
# 
# Cross-validation helps assess the model's generalization ability by evaluating it on different subsets of the data. It provides a more reliable estimate of the model's performance than a single train-test split.

# Calculating the Averege accuracy by taking the mean of these scores, which would give an overall estimate of the model's performance

# In[40]:


import numpy as np

accuracy_scores = np.array([0.82608696, 0.86956522, 0.84782609, 0.91304348, 0.84782609,
       0.80434783, 0.82608696, 0.77777778, 0.88888889, 0.68888889])

average_accuracy = np.mean(accuracy_scores)
print("Average accuracy:", average_accuracy)


# The average accuracy calculated from the given array of accuracy scores is approximately 0.8290 or 82.90%. This value represents the overall estimate of the model's performance across the different folds of the cross-validation process.

# ### Testing for accuracy

# In[41]:


result2.mean()*100


# This means that the model correctly predicted the target variable for approximately 82.90% of the instances in the testing dataset. It indicates the overall performance of the model on unseen data.

# ### Generating predictions with the model using the value of x test

# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat2=confusion_matrix(y_test,y_pred2)
conf_mat2


# Here's the breakdown of the confusion matrix:
# 
# True positive (TP): 79
# This represents the number of instances that are truly positive (belonging to class 1) and are correctly predicted as positive.
# 
# False positive (FP): 10
# This represents the number of instances that are actually negative (belonging to class 2) but are incorrectly predicted as positive.
# 
# False negative (FN): 28
# This represents the number of instances that are actually positive (belonging to class 1) but are incorrectly predicted as negative.
# 
# True negative (TN): 88
# This represents the number of instances that are truly negative (belonging to class 2) and are correctly predicted as negative
# 
# 

# ### visualizing confusion_matrix for training
# 

# In[43]:


ax=plt.subplot()
sns.heatmap(conf_mat2, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels([' cassava','yam']);
ax.yaxis.set_ticklabels(['yam', 'cassava']);


# ### Printing the Evaluation Metrics of the Model

# In[44]:


from sklearn.metrics import r2_score
MSE2 = np.square(np.subtract(y_test, y_pred2)).mean()
RMSE2 = math.sqrt(MSE1)
R_squared2 = r2_score(y_test, y_pred2)
print("Mean Square Error: ", MSE2)
print("Root Mean Square Error: ", RMSE2)
print("Coefficient of Determination: ", R_squared2)


# 

# 
# # RANDOM FOREST

# In[45]:


RFC=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)                          
rfc_model = RFC.fit(X_train,Y_train)


# ### MODEL TRAINING 

# In[46]:


y_pred_training = rfc_model.predict(X_train)
(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# ### MODEL TESTING 

# In[47]:


y_pred3 = rfc_model.predict(x_test)
(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))


# ### 10-f0ld cross validation of random forest classifier

# In[48]:


kfold = model_selection.KFold(n_splits=10, random_state =1, shuffle=True)
result3 = model_selection.cross_val_score(RFC, X_train, Y_train, cv=kfold)
result3
result3.mean()*100


# In[ ]:





# ### generating predictions with the model using the value of x test
# 

# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat3=confusion_matrix(y_test,y_pred3)
conf_mat3


# Here's the breakdown of the confusion matrix:
# 
# True positive (TP): 83
# This represents the number of instances that are truly positive (belonging to class 1) and are correctly predicted as positive.
# 
# False positive (FP): 6
# This represents the number of instances that are actually negative (belonging to class 2) but are incorrectly predicted as positive.
# 
# False negative (FN): 15
# This represents the number of instances that are actually positive (belonging to class 1) but are incorrectly predicted as negative.
# 
# True negative (TN): 93
# This represents the number of instances that are truly negative (belonging to class 2) and are correctly predicted as negative
# 
# 

# ### visualizing confusion_matrix for training

# In[50]:


ax=plt.subplot()
sns.heatmap(conf_mat3, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['cassava','yam']);
ax.yaxis.set_ticklabels(['yam','cassava'])


# ### Printing the Evaluation Metrics of the Model

# In[51]:


from sklearn.metrics import r2_score

MSE3 = np.square(np.subtract(y_test, y_pred3)).mean()
RMSE3 = math.sqrt(MSE3)
R_squared3 = r2_score(y_test, y_pred3)
	
print("Mean Square Error: ", MSE3)
print("Root Mean Square Error: ", RMSE3)
print("Coefficient of Determination: ", R_squared3)


# # Support vector machine

# In[52]:


LSVC=SVC(max_iter=1,kernel='linear', random_state=0)
LSVC_model = LSVC.fit(X_train,Y_train)


# ### SVM MODEL TRAININIG 

# In[53]:


y_pred_training = LSVC_model.predict(X_train)


# In[54]:


#print(classification_report(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# In[55]:


#print(confusion_matrix(Y_train,y_pred_training))
print(confusion_matrix(Y_train,y_pred_training))


# In[ ]:





# ### TESTING OF LSVM

# In[56]:


y_pred4 = LSVC_model.predict(x_test)
(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))


# ### 10-f0ld cross validation of linear support vector machine
# 

# In[57]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result4 = model_selection.cross_val_score(LSVC, X_train, Y_train, cv=kfold)
result4
result4.mean()*100


# ### generating predictions with the model using the value of x test

# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat4=confusion_matrix(y_test,y_pred4)
conf_mat4


# ### visualizing confusion_matrix for training

# In[59]:


ax=plt.subplot()
sns.heatmap(conf_mat4, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['cassava','yam']);ax.yaxis.set_ticklabels(['yam','cassava']);


# ### Printing the Evaluation Metrics of the Model

# In[60]:


from sklearn.metrics import r2_score

MSE4 = np.square(np.subtract(y_test, y_pred4)).mean()
RMSE4 = math.sqrt(MSE4)
R_squared4 = r2_score(y_test, y_pred4)

print("Mean Square Error: ", MSE4)
print("Root Mean Square Error: ", RMSE4)
print("Coefficient of Determination: ", R_squared4)


# # logistic regresssion model

# In[61]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
LogisticRegression()


# ### training the model

# In[62]:


y_pred_training12=logreg.predict(X_train)
(confusion_matrix(Y_train,y_pred_training12))
print(classification_report(Y_train,y_pred_training12))


# ### testing the model

# In[63]:


y_pred12=logreg.predict(x_test)
(confusion_matrix(y_test,y_pred12))
print(classification_report(y_test,y_pred12))


# ### generating predictions with the model using the value of x test

# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat12=confusion_matrix(y_test,y_pred12)
conf_mat12


# ### visualizing confusion_matrix for training

# In[65]:


ax=plt.subplot()
sns.heatmap(conf_mat12, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['yam','cassava']);ax.yaxis.set_ticklabels(['yam','cassava']);


# ### K-Fold validation 

# In[66]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result12 = model_selection.cross_val_score(logreg, X_train, Y_train, cv=kfold)
result12
result12.mean()*100


# ### Printing the Evaluation Metrics of the Model

# In[67]:


from sklearn.metrics import r2_score

MSE12 = np.square(np.subtract(y_test, y_pred12)).mean()
RMSE12 = math.sqrt(MSE12)
R_squared12 = r2_score(y_test, y_pred12)

print("Mean Square Error: ", MSE12)
print("Root Mean Square Error: ", RMSE12)
print("Coefficient of Determination: ", R_squared12)



# # Ensemble learning (Adaboost)

# In[68]:


from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(X_train,Y_train)
AdaBoostClassifier()
y_pred_training=adaboost.predict(X_train)
(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# ### testing the model

# In[69]:


y_pred10 =adaboost.predict(x_test)
(confusion_matrix(y_test,y_pred10))
print(classification_report(y_test,y_pred10))


# In[70]:


from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

# Create the voting classifier
voting = VotingClassifier(estimators=[('SVM', LSVC ), ('logistics Regression', logreg)])

# Define the KFold
kfold = model_selection.KFold(n_splits=10, random_state=0, shuffle=True)

# Perform cross-validation
result10 = model_selection.cross_val_score(voting, X_train, Y_train, cv=kfold)

# Display the results
print(result10)
print(result10.mean() * 100)


# The output you provided shows the cross-validation results for the VotingClassifier, with 10-fold cross-validation. The accuracy scores for each fold are displayed in the array:
# 
# [0.65217391, 0.5, 0.60869565, 0.60869565, 0.54347826, 0.58695652, 0.47826087, 0.48888889, 0.57777778, 0.64444444]
# 
# The mean accuracy across all folds is calculated by taking the average of these scores and multiplying it by 100:
# 
# 56.893719806763286
# 
# This means that, on average, the VotingClassifier achieved an accuracy of approximately 56.89% in the cross-validation process

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ### 10 fold cross validation of soft voting 

# In[71]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result10= model_selection.cross_val_score(voting, X_train, Y_train, cv=kfold)
result10
result10.mean ()*100


# ### generating predictions with the model using the value of x test

# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat10=confusion_matrix(y_test,y_pred10)
conf_mat10


# ### visualizing confusion_matrix for training

# In[73]:


ax=plt.subplot()
sns.heatmap(conf_mat10, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['yam','cassava']);ax.yaxis.set_ticklabels(['yam','cassava']);



# ### Printing the Evaluation Metrics of the Model

# In[74]:


from sklearn.metrics import r2_score

MSE10 = np.square(np.subtract(y_test, y_pred10)).mean()
RMSE10 = math.sqrt(MSE10)
R_squared10 = r2_score(y_test, y_pred10)

print("Mean Square Error: ", MSE10)
print("Root Mean Square Error: ", RMSE10)
print("Coefficient of Determination: ", R_squared10)


# 
# # Hard voting
# 

# ### combining decision tree, randomforest and knn
# 

# In[75]:


votingh = VotingClassifier(estimators = [("DTC",dtc_model), 
                                        ("rf",rfc_model),
                                        ("knn",knn_model)],  
                          voting = 'hard',
                          weights=None,
                          n_jobs=None,
                          flatten_transform=True,)


# ### training the model

# In[76]:


votingh_model = voting.fit(X_train,Y_train)
y_pred_training = votingh_model.predict(X_train)
(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))
 


# ### testing the model
# 

# In[77]:


y_pred8 =votingh_model.predict(x_test)
(confusion_matrix(y_test,y_pred8))
print(classification_report(y_test,y_pred8))
 


# ### 10-fold cross validation of hard voting 

# In[78]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result8 = model_selection.cross_val_score(voting, X_train, Y_train, cv=kfold)
result8
result8.mean()*100


# ### generating predictions with the model using the value of x test
# 

# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat8=confusion_matrix(y_test,y_pred8)
conf_mat8


# ### visualizing confusion_matrix for training
# 

# In[80]:


ax=plt.subplot()
sns.heatmap(conf_mat8, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['cassava','yam']);ax.yaxis.set_ticklabels(['yam','cassava']);


# ### Printing the Evaluation Metrics of the Model

# In[81]:


from sklearn.metrics import r2_score

MSE8 = np.square(np.subtract(y_test, y_pred8)).mean()
RMSE8 = math.sqrt(MSE8)
R_squared8 = r2_score(y_test, y_pred8)
print("Mean Square Error: ", MSE8)
print("Root Mean Square Error: ", RMSE8)
print("Coefficient of Determination: ", R_squared8)



# # Majority voting (soft)
# 

# ### combining decision tree, randomforest and knn
# 

# In[82]:


voting = VotingClassifier(estimators = [("DTC",dtc_model), 
                                        ("rf",rfc_model),
                                        ("knn",knn_model)],  
                          voting = 'soft',
                          weights=None,
                          n_jobs=None,
                          flatten_transform=True,)
voting_model = voting.fit(X_train,Y_train)
y_pred_training = voting_model.predict(X_train)
(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# ### testing the model

# In[83]:


y_pred9 =voting_model.predict(x_test)
(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))


# ### 10 fold cross validation of soft voting 

# In[84]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result9 = model_selection.cross_val_score(voting, X_train, Y_train, cv=kfold)
result9
result9.mean()*100


# ### generating predictions with the model using the value of x test

# In[85]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat9=confusion_matrix(y_test,y_pred9)
conf_mat9


# ### visualizing confusion_matrix for training

# In[86]:


ax=plt.subplot()
sns.heatmap(conf_mat9, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['cassava','yam']);ax.yaxis.set_ticklabels(['yam','cassava']);


# ### Printing the Evaluation Metrics of the Model

# In[87]:


from sklearn.metrics import r2_score

MSE9 = np.square(np.subtract(y_test, y_pred9)).mean()
RMSE9 = math.sqrt(MSE9)
R_squared9 = r2_score(y_test, y_pred9)
print("Mean Square Error: ", MSE9)
print("Root Mean Square Error: ", RMSE9)
print("Coefficient of Determination: ", R_squared9)


# 
# # STACK CLASSIFIER
# 

# In[88]:


from sklearn.ensemble import StackingClassifier
level0=list()
level0.append(('knn',knn_model))
level0.append(('dtc',DecisionTreeClassifier()))
level0.append(('rf',RFC))

level1 = KNeighborsClassifier(n_neighbors=200, metric="minkowski")
stacking = StackingClassifier(estimators=level0,final_estimator=level1)
stacking.fit(X_train,Y_train)
StackingClassifier(estimators=[('knn', KNeighborsClassifier()),
                               ('dtc', DecisionTreeClassifier()),
                               ('rf',
                                RandomForestClassifier(criterion='entropy',
                                                       n_estimators=10,
                                                       random_state=0))],
                   final_estimator=KNeighborsClassifier(n_neighbors=200))
y_pred_training =stacking.predict(X_train)
(confusion_matrix(Y_train,y_pred_training))
print(classification_report(Y_train,y_pred_training))


# 
# ### testing the model 
# 

# In[89]:


y_pred7 =stacking.predict(x_test)

(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))
 


#              
# ### 10-f0ld cross validation
# 

# In[90]:


kfold = model_selection.KFold(n_splits=10, random_state =0, shuffle=True)
result7 = model_selection.cross_val_score(stacking, X_train, Y_train, cv=kfold)
result7
result7.mean()*100


# 
# ### generating predictions with the model using the value of x test
# 

# In[91]:


import seaborn as sns
import matplotlib.pyplot as plt
conf_mat7=confusion_matrix(y_test,y_pred7)
conf_mat7


# 
# ### visualizing confusion_matrix for training
# 

# In[92]:


ax=plt.subplot()
sns.heatmap(conf_mat7, annot=True, ax=ax)

ax.set_xlabel('predicted labels');ax.set_ylabel('actual labels');
ax.set_title('confusion matrix');
ax.xaxis.set_ticklabels(['cassava','yam']);ax.yaxis.set_ticklabels(['yam','cassava']);


# In[93]:


from sklearn.metrics import r2_score
MSE2 = np.square(np.subtract(y_test, y_pred2)).mean()
RMSE2 = math.sqrt(MSE1)
R_squared2 = r2_score(y_test, y_pred2)
print("Mean Square Error: ", MSE2)
print("Root Mean Square Error: ", RMSE2)
print("Coefficient of Determination: ", R_squared2)


# # Section of graphical representation of the results

# In[94]:


#library
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ### creating the dataset

# In[95]:


x,y=make_classification()
X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)
data = {'KNN':68, 'DT':83, 'RF':85, 'SVM':55,'ABT':57,'Hard_V1':85,'Soft_V2':86,'LR':61,'ST':85}
Models = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (15, 10))


# ### creating the bar plot
# 

# In[95]:


plt.bar(Models, values, color =['blue','green','orange','yellow','purple','brown','red','maroon','magenta','black'], alpha=0.7)
plt.grid(color='#95a5a6',linestyle='--',linewidth=2, axis='y', alpha=0.7)
plt.xlabel("models used")
plt.ylabel("Percentage accuracy %")
plt.show()


# ### bar graph representation of accuracy
# 

# In[96]:


import matplotlib.pyplot as plt
data = {'KNN':68, 'DT':83, 'RF':85, 'SVM':55,'ABT':57,'Hard_V1':85,'Soft_V2':86,'LR':61,'ST':85}
Models = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (15, 10))

plt.bar(Models, values)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()



# In[97]:


import matplotlib.pyplot as plt

data = {'KNN':68, 'DT':83, 'RF':85, 'SVM':55,'ABT':57,'Hard_V1':85,'Soft_V2':86,'LR':61,'ST':85}
Models = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(15, 10))
plt.bar(Models, values)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()


# ### creating the bar plot

# In[98]:


plt.bar(Models, values, color ='g', alpha=0.7)
plt.grid(color='#95a5a6',linestyle='--',linewidth=2, axis='y', alpha=0.7)
plt.xlabel("models used")
plt.ylabel("Percentage accuracy %")
plt.show()


# ### Pair plot of dataset indicating distribution of data points 

# In[99]:


import pandas as pd
import seaborn as sns
import seaborn 
import matplotlib.pyplot as plt
dataset=pd.read_csv("dataset1.csv")
plt.figure()
seaborn.pairplot(dataset[['MAX_TEMP','RH60','Rainfall']],)
plt.show()


# ### Plot of precision and recall

# In[100]:


import numpy as np
import matplotlib.pyplot as plt 
X=['KNN', 'DT', 'RF', 'SVM','LR','ABT','V1','V2','ST']
x_Pricision = [74, 80, 85, 51, 52, 70, 83, 85, 93]
y_f1_score = [70, 84, 89, 64, 49, 75, 87, 89, 94]

X_axix=np.arange(len(X))
plt.bar(X_axix - 0.2, x_Pricision, 0.4,label='pricision', color='r')
plt.bar(X_axix + 0.2, y_f1_score, 0.4,label='f1 score', color='g',)
plt.xticks(X_axix, X)
plt.xlabel("ALGORITHMS")
plt.ylabel('Evaluation parameters')
plt.title("Class 1 Performance Metrics")
plt.legend()
plt.show()


# In[101]:


import numpy as np
import matplotlib.pyplot as plt 
X=['KNN', 'DT', 'RF', 'SVM','LR','ABT','V1','V2','ST']
x_Pricision = [74, 90, 94, 74, 59, 82, 93, 95, 96]
y_f1_score = [77, 85, 90, 45, 62, 76, 88, 90, 95]


X_axix=np.arange(len(X))
plt.bar(X_axix - 0.2, x_Pricision, 0.4,label='pricision', color='r')
plt.bar(X_axix + 0.2, y_f1_score, 0.4,label='f1 score', color='g',)
plt.xticks(X_axix, X)
plt.xlabel("ALGORITHMS")
plt.ylabel('Evaluation parameters')
plt.title("Class 2 Performance Metrics")
plt.legend()
plt.show()


# ### Plot of AUC and ROC

# In[102]:


import math 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
probs=model.predict_proba(x_test)
preds=probs[:, 1]
fpr, tpr, threshhold = metrics.roc_curve(y_test, preds)
roc_auc=metrics.auc(fpr,tpr)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics



# ### define the predictor variables and response variables

# In[103]:


x=dataset[['Rainfall','MAX_TEMP','RH60']]
y=dataset['CROP']



# ### AUC AND ROC Logistic Regression 
# instantiate the models

# In[105]:


model= LogisticRegression()


# fit the model uusing the training data

# In[106]:


model.fit(X_train,Y_train)
LogisticRegression()



# Define metrics

# In[107]:


y_pred_proba=model.predict_proba(x_test)[::,1]
fpr, tpr, threshhold = metrics.roc_curve(y_test, y_pred_proba)


# Create ROC curve

# In[108]:


plt.title('Receiver Operatinng Characteristic ROC')
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Estimating area under curve AUC

# In[109]:


auc=metrics.roc_auc_score(y_test, y_pred_proba)


# ### ROC and AUC for Random Forest 
# Import libraries

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ROC curve and AUC score

# In[111]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Defining a python function to plot the ROC curves
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label="AUC= " +str(auc))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
#Generate sample data.
x,y=make_classification()
#Split the data into train and test sub-datasets
X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)

#Fit a model on the train data.
model1 = RandomForestClassifier()
model1.fit(X_train,Y_train)
RandomForestClassifier()

#Predict probabilities for the test data.
probs = model1.predict_proba(x_test)

#Keep Probabilities of the positive class only
probs = probs[:, 1]

#Compute the AUC Score
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

#Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

#Plot ROC Curve using the defined function
plot_roc_curve(fpr, tpr)


# 

# In[112]:


#Generate sample data.
x,y=make_classification()
#Split the data into train and test sub-datasets
X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)

#Fit a model on the train data.
model1 = RandomForestClassifier()
model1.fit(X_train,Y_train)
RandomForestClassifier()

#Predict probabilities for the test data.
probs = model1.predict_proba(x_test)

#Keep Probabilities of the positive class only
probs = probs[:, 1]

#Compute the AUC Score
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

#Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

#Plot ROC Curve using the defined function
plot_roc_curve(fpr, tpr)


# ## AUC AND ROC FOR DECISION TREE

# In[113]:


#Defining a python function to plot the ROC curves
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label="AUC= " +str(auc))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

#Generate sample data.
x,y=make_classification()

#Split the data into train and test sub-datasets
X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)

#Fit a model on the train data.
dtc= DecisionTreeClassifier(criterion = "entropy", min_samples_leaf=2,
                            random_state = 100)
dtc_model = dtc.fit(X_train,Y_train)

#Predict probabilities for the test data.
probs = dtc.predict_proba(x_test)

#Keep Probabilities of the positive class only
probs = probs[:, 1]

#Compute the AUC Score
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
#Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

#Plot ROC Curve using our defined function
plot_roc_curve(fpr, tpr)


# ## AUC and ROC for AdaBoot classifier

# In[114]:


#Defining a python function to plot the ROC curves
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label="AUC= " +str(auc))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

#Generate sample data.
x,y=make_classification()

#Split the data into train and test sub-datasets
X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size =0.3, random_state =0)

#Fit a model on the train data.
adaboost = AdaBoostClassifier()
adaboost.fit(X_train,Y_train)

AdaBoostClassifier()

#Predict probabilities for the test data.
probs = adaboost.predict_proba(x_test)

#Keep Probabilities of the positive class only
probs = probs[:, 1]

#Compute the AUC Score
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

#Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

#Plot ROC Curve using our defined function
plot_roc_curve(fpr, tpr)


# ## The code below saves the model as a pickle file

# In[117]:


import joblib
# Save each model using joblib
joblib.dump(dtc_model, 'decision.pkl')
joblib.dump(knn_model, 'Knn.pkl')
joblib.dump(votingh_model, 'Hard.pkl')
joblib.dump(voting_model, 'Soft.pkl')
joblib.dump(rfc_model, 'Random.pkl')
joblib.dump(stacking, 'Stacking.pkl')


# In[118]:


import os

model_path = os.path.abspath('model.pkl')
print(f"The saved model is located at: {model_path}")


# In[119]:


pip install flask


# In[ ]:





# In[77]:


import joblib
joblib.dump(rfc_model, 'model.pkl')

