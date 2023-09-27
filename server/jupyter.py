#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# libraries for computation
import pandas as pd
import numpy as np
#library for train test split
from sklearn.model_selection import train_test_split,cross_val_score,KFold

#library for preprocessing
from sklearn.preprocessing import StandardScaler

#library for Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#Library for feature selection techniques
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

#libraries for various ML models 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#ensemble models
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor

#libraries for model performance evaluation
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

#libraries for visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings('ignore')


# In[117]:


data = pd.read_csv("finaldata.csv")
data.head()


# In[118]:


y = pd.read_csv("finaltest.csv")
y.head()


# In[119]:


# check for missing values in the dataframe
print(data.isnull().values.any())


# In[120]:


# check for missing values in the testframe(optional)
print(y.isnull().values.any())


# In[121]:


#Descriptive Statistics
data.describe()


# In[122]:


print("Shape of data: {}".format(data.shape))


# In[123]:


print("Shape of y: {}".format(y.shape))


# In[124]:


#class distribution(classification only)
Target_counts=data.groupby('Target').size()
print(Target_counts)


# In[125]:


#Pairwise Pearson correlations
from pandas import set_option
set_option('display.width',100)
# set_option('precision', 3)
correlations=data.corr(method='pearson')
print(correlations)


# In[126]:


data.columns


# In[127]:


#Skew for each attribute
skew=data.iloc[:,1:32].skew()
print(skew)


# In[128]:


data['user'].value_counts()


# In[129]:


data['Target'].value_counts()


# In[130]:


#Univariate Histograms
# data.iloc[214:269,1:32].hist(sharex=False,bins=4,figsize=(15,15),sharey=False, xlabelsize=2,ylabelsize=2)
data.iloc[214:269,1:32].hist(sharex=False,figsize=(15,15),sharey=False, xlabelsize=2,ylabelsize=2)
plt.show()


# In[131]:


#Generate a sequence of numbers from-10 to10 with 100 steps inbetween
x=data.iloc[214:269, 1:32]
x=np.linspace(-10,10,100)
#Create a second array using sine
y=np.sin(x)
#The plot function makes a line chart of one array against another
plt.plot(x,y,marker="x")


# In[132]:


# # Slice the data
# data_slice = data.iloc[214:269, 1:32]

# # Create a figure and a grid of subplots
# fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(30,30))

# # Flatten the axes array
# axes = axes.flatten()

# # Remove extra subplots
# for i in range(len(data_slice.columns), len(axes)):
#     fig.delaxes(axes[i])

# # Plot each column
# for i, col_name in enumerate(data_slice.columns):
#     sns.kdeplot(data_slice[col_name], ax=axes[i], fill=True)
#     axes[i].set_title(col_name, fontsize=12)
#     axes[i].set_xlabel('')
#     axes[i].set_ylabel('')
#     plt.tight_layout()
# plt.show()


# In[133]:


# import seaborn as sns
# #create density plot of data
# sns.kdeplot(data.iloc[214:269,1:32])
# plt.show()


# In[134]:


# #Univariate Density Plots
# data.iloc[214:269,1:32].plot(kind='density', subplots=True, figsize=(80,80),layout=(8,8), sharex=2, legend=False,fontsize=9)
# plt.show()


# In[135]:


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the data
# filename = "finaldata.csv"
# data = pd.read_csv(filename)

# # Compute the correlation matrix
# corr = data.iloc[214:269,1:32].corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

# plt.show()


# In[136]:


# #Box and Whisker Plots
# data.iloc[214:269,1:32].plot(kind='box',figsize=(14, 14),subplots=True,layout=(6,6),sharex=False,sharey=False)
# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.7, wspace=0.9)
# plt.show()


# In[137]:


#Multivariate Plots
#correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
plt.show()


# In[138]:


#Multivariate Plots
#correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.iloc[214:269,1:32].corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
plt.show()


# In[139]:


# #Scatterplot.Matrix
# from pandas.plotting import scatter_matrix
# scatter_matrix(data.iloc[214:269,1:32],figsize=(90, 90), diagonal='hist')
# # Set the font size for tick labels
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.show()


# In[140]:


sns.countplot(x='Target',data=data,palette='RdBu_r')


# In[141]:


data = pd.read_csv("finaldata.csv")
y = pd.read_csv("finaltest.csv")

features = list(data.columns[1:32])
# X = data[features]
# y = data['Target']
# train, test = train_test_split(data, test_size = 0.2)
X_train = data[features]
y_train = data['Target']
# X_test = test[features]
# y_test = test['Target']
features = list(y.columns[1:32])
X_test = y[features]
y_test = y['Target']


# In[142]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)


# In[143]:


def plotConfusion(cm):
    sns.set_style('white')
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Pastel1)
    classNames = ['Genuine','Imposter']
    plt.title('Confusion Matrix',fontsize = 15)
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames,fontsize=12)
    plt.yticks(tick_marks, classNames,fontsize=12)
    s = [['TP','FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[144]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
pred = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test,pred)
print(cm)
plotConfusion(cm)


# In[145]:


#KNN for different k values
kVals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
f1_scores = []

for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled,y_train)
    pred = knn.predict(X_test_scaled)
    f1_scores.append(f1_score(y_test,pred,average='weighted'))
    
plt.plot(kVals,f1_scores)    
print(f1_scores) 


# In[146]:


from sklearn.metrics import roc_curve, auc
# Predict probabilities
y_prob = knn.predict_proba(X_test_scaled)
#print(y_prob)
print(len(y_test))
print(len(y_prob))
# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Fit a model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
knn.fit(X_train_scaled, y_train)


# In[147]:


# Get the model's probabilities for the positive class
# y_scores = model.predict_proba(X_test)[:, 1]
y_scores = knn.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()



# In[148]:


from sklearn.metrics import accuracy_score
pred = knn.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)



# In[149]:


#Logistic Regression
logmodel = LogisticRegression()
logmodel.fit(X_train_scaled,y_train)
predtest = logmodel.predict(X_test_scaled)
predtrain = logmodel.predict(X_train_scaled)

print("F1 Score: ", metrics.f1_score(y_train,predtrain,average='weighted'))
print("F1 Score: ", metrics.f1_score(y_test,predtest,average='weighted'))

print("Training Accuracy Score: ", accuracy_score(y_train,predtrain))
print("Testing Accuracy Score: ", accuracy_score(y_test,predtest))


# In[150]:


# Predict probabilities
y_prob = logmodel.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[151]:


# Fit a model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
logmodel.fit(X_train_scaled, y_train)

# Get the model's probabilities for the positive class
# y_scores = model.predict_proba(X_test)[:, 1]
y_scores = knn.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 
print(fpr)
print(np.absolute(fnr - fpr))
print(np.nanargmin(np.absolute((fnr - fpr))))
# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[152]:


# Fit a model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[153]:


# Get the model's probabilities for the positive class
y_scores = model.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[154]:


pred = logmodel.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[155]:


#DT
d_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
d_classifier.fit(X_train_scaled,y_train)
pred = d_classifier.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[156]:


# Predict probabilities
y_prob = d_classifier.predict_proba(X_test_scaled)
#lculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[157]:


# Fit a model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
d_classifier.fit(X_train_scaled, y_train)

# Get the model's probabilities for the positive class
# y_scores = model.predict_proba(X_test)[:, 1]
y_scores = knn.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 
# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[158]:


# Get the model's probabilities for the positive class
y_scores = d_classifier.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[159]:


pred = d_classifier.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[160]:


#Ensemble Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state = 42)
rf_classifier.fit(X_train_scaled,y_train)
pred = rf_classifier.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[161]:


# Predict probabilities
y_prob = rf_classifier.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[162]:


# Get the model's probabilities for the positive class
y_scores = rf_classifier.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[163]:


pred = rf_classifier.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[164]:


#Ensemble Random Forest Classifier Fine Tunning
estimators = [5,10,15,20,30,35,40,48]
f1_scores = []
for e in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=e, random_state = 42)
    rf_classifier.fit(X_train_scaled,y_train)
    pred = rf_classifier.predict(X_test_scaled)
    f1_scores.append(f1_score(y_test,pred,average='weighted'))
    
plt.plot(estimators,f1_scores) 
print(f1_scores)


# In[165]:


# Get the model's probabilities for the positive class
y_scores = rf_classifier.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[166]:


pred = rf_classifier.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[167]:


#Bagging Classifier with Decision tree as base learner
cart = DecisionTreeClassifier()
cart.fit(X_train_scaled, y_train)
model = BaggingClassifier(base_estimator=cart, n_estimators=150, random_state=7)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[168]:


# Predict probabilities
y_prob = cart.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[169]:


# Get the model's probabilities for the positive class
y_scores = cart.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[170]:


# Get the model's probabilities for the positive class
y_scores = model.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[171]:


model = BaggingClassifier(base_estimator=cart, n_estimators=150, random_state=7)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[172]:


model = BaggingClassifier(base_estimator=cart, n_estimators=150, random_state=7)
model.fit(X_train_scaled, y_train)
# Predict probabilities
y_prob = model.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()


# In[173]:


# Get the model's probabilities for the positive class
y_scores = model.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[174]:


num_trees = 80
max_features = 30
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[175]:


model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
model.fit(X_train_scaled, y_train)
# Predict probabilities
y_prob = model.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[176]:


# Get the model's probabilities for the positive class
y_scores = model.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[177]:


# AdaBoostClassifier
seed = 7
num_trees = 100
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[178]:


model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
# Predict probabilities
y_prob =model.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[179]:


# Fit a model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
model.fit(X_train_scaled, y_train)


# In[180]:


# Get the model's probabilities for the positive class
# y_scores = model.predict_proba(X_test)[:, 1]
y_scores = knn.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[181]:


model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[182]:


seed = 7
num_trees = 100
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[183]:


model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
# Predict probabilities
y_prob = model.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[184]:


# Get the model's probabilities for the positive class
y_scores =model.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[185]:


model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[186]:


# Try with Different Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
# create the sub models
estimators = []
model1 = KNeighborsClassifier(n_neighbors=5)
estimators.append(('Knn', model1))
model2 = RandomForestClassifier(n_estimators=100, max_features=30)
estimators.append(('RandomForest', model2))
model3 = ExtraTreesClassifier(n_estimators=100, max_features=30)
estimators.append(('ExtraTree', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train_scaled, y_train)
pred = ensemble.predict(X_test_scaled)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))
cm = confusion_matrix(y_test,pred)

print(cm)
plotConfusion(cm)
# voting_classifier = VotingClassifier(estimators=classifiers, voting='soft')


# In[187]:


# Predict probabilities
y_prob = ensemble.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[188]:


# Get the model's probabilities for the positive class
y_scores = ensemble.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[189]:


pred = ensemble.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[190]:


#MLPClassifier
net = net = MLPClassifier(random_state=2,hidden_layer_sizes=(100,200,330,10),max_iter=500,activation= 'relu', learning_rate= 'invscaling', solver='adam')
net.fit(X_train_scaled,y_train)
pred= net.predict(X_test_scaled)

print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[191]:


# Predict probabilities
y_prob = net.predict_proba(X_test_scaled)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()


# In[192]:


# Get the model's probabilities for the positive class
y_scores = net.predict_proba(X_test_scaled)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC curve and EER')
plt.legend()
plt.show()



# In[193]:


net = net = MLPClassifier(random_state=2,hidden_layer_sizes=(100,200,330,10),max_iter=500,activation= 'relu', learning_rate= 'invscaling', solver='adam')
net.fit(X_train_scaled,y_train)
pred= net.predict(X_test_scaled)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[194]:


pca = PCA(n_components=15)  
X_train_pca = pca.fit_transform(X_train_scaled)  
X_test_pca = pca.transform(X_test_scaled)


# In[195]:


#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca,y_train)
pred = knn.predict(X_test_pca)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[196]:


pred= knn.predict(X_test_pca)
# Assuming you have true labels 'y_test' and predicted labels 'y_pred'
testing_accuracy = accuracy_score(y_test, pred)

print("Testing Accuracy:", testing_accuracy)


# In[197]:


# Predict probabilities
y_prob = knn.predict_proba(X_test_pca)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[198]:


# Get the model's probabilities for the positive class
y_scores = knn.predict_proba(X_test_pca)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[199]:


#KNN for different k values
kVals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
f1_scores = []

for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca,y_train)
    pred = knn.predict(X_test_pca)
    f1_scores.append(f1_score(y_test,pred,average='weighted'))
    
plt.plot(kVals,f1_scores)    
print(f1_scores) 


# In[200]:


#Ensemble Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=150, random_state = 42)
rf_classifier.fit(X_train_pca,y_train)
pred = rf_classifier.predict(X_test_pca)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[201]:


# Predict probabilities
y_prob = knn.predict_proba(X_test_pca)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[202]:


# Get the model's probabilities for the positive class
y_scores =rf_classifier.predict_proba(X_test_pca)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[203]:


#ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100, max_features=5)
model.fit(X_train_pca, y_train)
pred = model.predict(X_test_pca)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[204]:


# Predict probabilities
y_prob = model.predict_proba(X_test_pca)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[205]:


# Get the model's probabilities for the positive class
y_scores =model.predict_proba(X_test_pca)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[206]:


pca = PCA(n_components=5)  
X_train_pca = pca.fit_transform(X_train_scaled)  
X_test_pca = pca.transform(X_test_scaled) 
#Version 13 imporvement
# Try with Different Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
# create the sub models
estimators = []
model1 = KNeighborsClassifier(n_neighbors=4)
estimators.append(('Knn', model1))
model2 = RandomForestClassifier(n_estimators=100, max_features=5)
estimators.append(('RandomForest', model2))
model3 = ExtraTreesClassifier(n_estimators=100, max_features=5)
estimators.append(('ExtraTree', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators,voting='soft')
ensemble.fit(X_train_pca, y_train)
pred = ensemble.predict(X_test_pca)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[207]:


# Get the model's probabilities for the positive class
y_scores =ensemble.predict_proba(X_test_pca)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[208]:


# Predict probabilities
y_prob = ensemble.predict_proba(X_test_pca)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[209]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
scores=cross_val_score(logreg,X_train_pca,y_train,cv=10)
print("Cross-validation scores:{}".format(scores))
print("Average cross-validation score:{:.2f}".format(scores.mean()))
import matplotlib.pyplot as plt

# Create a range for x axis
x = range(1, len(scores) + 1)

# Create bar chart
plt.bar(x, scores)
plt.xlabel('Fold number')
plt.ylabel('Cross-validation score')
plt.title('Cross-validation scores bar chart')
plt.show()


# In[210]:


svd = TruncatedSVD(n_components=31)
X_train_svd = svd.fit_transform(X_train_scaled,y_train)
X_test_svd = svd.transform(X_test_scaled)


# In[211]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
scores=cross_val_score(logreg,X_train_svd,y_train,cv=10)
print("Cross-validation scores:{}".format(scores))
print("Average cross-validation score:{:.2f}".format(scores.mean()))
import matplotlib.pyplot as plt

# Create a range for x axis
x = range(1, len(scores) + 1)

# Create bar chart
plt.bar(x, scores)
plt.xlabel('Fold number')
plt.ylabel('Cross-validation score')
plt.title('Cross-validation scores bar chart')
plt.show()


# In[212]:


#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_svd,y_train)
pred = knn.predict(X_test_svd)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[213]:


# Predict probabilities
y_prob = knn.predict_proba(X_test_svd)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[214]:


# Get the model's probabilities for the positive class
y_scores =knn.predict_proba(X_test_svd)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[215]:


#KNN for different k values
kVals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
f1_scores = []

for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_svd,y_train)
    pred = knn.predict(X_test_svd)
    f1_scores.append(f1_score(y_test,pred,average='weighted'))
    
plt.plot(kVals,f1_scores)    
print(f1_scores)


# In[216]:


#Ensemble Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=150, random_state = 42)
rf_classifier.fit(X_train_svd,y_train)
pred = rf_classifier.predict(X_test_svd)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[217]:


# Predict probabilities
y_prob =rf_classifier.predict_proba(X_test_svd)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[218]:


# Get the model's probabilities for the positive class
y_scores =rf_classifier.predict_proba(X_test_svd)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')
plt.legend()
plt.show()


# In[219]:


#ExtraTreeClassifier
model = ExtraTreesClassifier(n_estimators=50, max_features=20,random_state=7)
model.fit(X_train_svd, y_train)
pred = model.predict(X_test_svd)
print("F1 Score: ", metrics.f1_score(y_test,pred, average='weighted'))
print("Accuracy Score: ", accuracy_score(y_test,pred))


# In[220]:


# Predict probabilities
y_prob =model.predict_proba(X_test_svd)

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[221]:


# Get the model's probabilities for the positive class
y_scores =model.predict_proba(X_test_svd)[:, 1]


# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)

# Compute EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] 
print('EER: ', EER)
print('EER Threshold: ', eer_threshold)
# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.scatter(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], color='red')
plt.text(fpr[np.nanargmin(np.absolute((fnr - fpr)))],1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))], f'EER = {EER:.2f}', color='red', ha='right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve and EER')

plt.legend()
plt.show()


# In[222]:


# st="0.119,0.225,0.106,0.095,0.192,0.097,0.111,0.216,0.105,0.119,0.256,0.137,0.095,1.159,1.064,0.183,0.145,-0.038,0.127,0.632,0.505,0.103,0.248,0.145,0.119,0.198,0.104,-0.094,0.151,0.152,0.121"
# st="0.1,1.28,1.18,0.099,2.116,2.017,0.103,1.485,1.382,0.077,1.836,1.759,0.075,2.759,2.684,0.178,1.457,1.279,0.095,1.722,1.627,0.097,2.012,1.915,0.086,0.876,0.79,0.084,0.923,0.839,0.08"
# st="0.093,0.448,0.355,0.086,0.336,0.25,0.125,0.283,0.158,0.099,0.368,0.269,0.1,0.457,0.357,0.097,0.56,0.463,0.095,0.191,0.096,0.142,0.22,0.078,0.080,.325,0.245,0.076,0.327,0.251,0.098"
st="0.126,0.256,0.13,0.111,0.216,0.105,0.119,0.232,0.113,0.135,1.328,1.193,0.103,0.727,0.624,0.215,0.185,-0.03,0.111,0.16,0.049,0.135,0.2,0.065,0.143,0.216,0.073,0.239,0.096,-0.143,0.207"
#Genuine
# st="2.555,2.557,0.002,0.003,0.003,0,0.001,0.001,0,0.002,0.002,0,0.623,0.626,0.003,0.002,0.004,0.002,0.003,0.005,0.002,0.004,0.006,0.002,0.002,0.002,0,0.001,0.001,0,0"
#Imposter


# In[223]:


li = list(st.split(','))


# In[224]:


# res =ensemble.predict(arr)
li = list(st.split(','))


# In[225]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train)


# In[226]:


ar = np.array(li, dtype=float)
arr = ar.reshape(1, -1)
arr = scaler.transform(arr)

res = classifier.predict(arr)


# In[227]:


res[0]


# In[228]:


# import pandas as pd
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier

# # Load your data
# train_data = pd.read_csv('finaldata.csv')
# test_data = pd.read_csv('finaltest.csv')

# # Separate features and target
# X_train = train_data.drop('tAEGET', axis=1)
# y_train = train_data['tAEGET']
# X_test = test_data.drop('tAEGET', axis=1)
# y_test = test_data['tAEGET']

# Create and train a classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Get prediction scores
y_scores = clf.predict_proba(X_test)[:, 1]


# Compute ROC curve (False Positive Rate (FPR) and True Positive Rate (TPR)) 
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Compute FRR and FAR
far = fpr
frr = 1 - tpr

# Find where far = frr
eer_threshold = thresholds[abs(far - frr).argmin()]
eer = frr[abs(far - frr).argmin()]

print('EER: ', eer)
print('EER Threshold: ', eer_threshold)

# Plot the EER curve
plt.figure()
plt.plot(far, frr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Acceptance Rate')
plt.ylabel('False Rejection Rate')
plt.title('EER Tradeoff Curve')
plt.legend(loc="lower right")
plt.show()


# In[229]:


from scipy.stats import ttest_ind
# Separate the data into two groups based on the target label
Imposter_data = data[data['Target']  == 0]
Genuine_data = data[data['Target']  ==1]
# Perform t-tests for each column of data
for col in data.columns[1:32]:
    t, p = ttest_ind(Imposter_data[col], Genuine_data[col])
    print('Column:', col)
    print('t-statistic:', t)
    print('p-value:', p)


# In[230]:


# Calculate the coefficient of variation for each feature
cv = np.std(data, axis=0) / np.mean(data, axis=0)

# Rank the features by their coefficient of variation
rank = np.argsort(cv)[::-1]

# Display the features in order of distinctiveness with rank and score value
features = data.columns
for i, f in enumerate(features[rank]):
    print(f"Rank {i+1}: {f} ({cv[rank][i]:.2f})")


# In[231]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is a pandas DataFrame
cv = np.std(data, axis=0) / np.mean(data, axis=0)

# Rank the features by their coefficient of variation in descending order
ranks = np.argsort(cv)[::-1]

# Display the top 15 features in order of highest coefficient of variation with rank and score value
features = data.columns
for i, f in enumerate(features[ranks][:15]):
    print(f"Rank {i+1}: {f} ({cv[ranks][i]:.2f})")

# Create a bar plot
x_pos = np.arange(len(cv[ranks][:15]))
plt.figure(figsize=(12, 6))
plt.bar(x_pos, cv[ranks][:15], align='center', alpha=0.5, width=.70)
plt.xticks(x_pos, features[ranks][:15], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Coefficient of Variation')
plt.title('Top 15 Features Based on Highest Coefficient of Variation')
# Show the plot
plt.show()


# In[232]:


cv = np.std(data, axis=0) / np.mean(data, axis=0)
# Rank the features by their coefficient of variation
rank = np.argsort(cv)

# Display the top 15 features in order of lowest coefficient of variation with rank and score value
features = data.columns
for i, f in enumerate(features[rank][:15]):
    print(f"Rank {i+1}: {f} ({cv[rank][i]:.2f})")

# Create a bar plot
x_pos = np.arange(len(cv[rank][:15]))
plt.figure(figsize=(12, 6))
plt.bar(x_pos, cv[rank][:15], align='center', alpha=0.5, width=.70)
plt.xticks(x_pos, features[rank][:15], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Coefficient of Variation')
plt.title('Top 15 Features Based on Lowest Coefficient of Variation')

# Show the plot
plt.show()


# In[233]:


# Calculate the coefficient of variation for each feature
data=data.iloc[214:361,1:32]

cv = np.std(data, axis=0) / np.mean(data, axis=0)

# Rank the features by their coefficient of variation
rank = np.argsort(cv)[::-1]

# Display the features in order of distinctiveness with rank and score value
features = data.columns
for i, f in enumerate(features[rank]):
    print(f"Rank {i+1}: {f} ({cv[rank][i]:.2f})")
    
# #Create a bar plot
x_pos = np.arange(len(cv[rank]))
plt.figure(figsize=(12, 6))
plt.bar(x_pos, cv[rank], align='center', alpha=0.5, width=.70)
plt.xticks(x_pos, features[rank], rotation=65)
plt.xlabel('Feature')
plt.ylabel('Coefficient of Variation')
plt.title('Feature Distinctiveness Based on Coefficient of Variation')

# # Show the plot
plt.show()


# In[234]:


from scipy.spatial.distance import cdist

data = pd.read_csv('finaldata.csv')

# Separate Imposter and Genuine data
Imposter_data = data[data['Target'] == 0]
Genuine_data = data[data['Target'] == 1]

# Create templates
imposter_template = np.mean(Imposter_data.values[:, 1:32].astype(float), axis=0)
genuine_template = np.mean(Genuine_data.values[:, 1:32].astype(float), axis=0)

# Compute distances
imposter_distances = cdist(imposter_template.reshape(1, -1), data.values[:, 1:32].astype(float), metric='euclidean')
genuine_distances = cdist(genuine_template.reshape(1, -1), data.values[:, 1:32].astype(float), metric='euclidean')

# Compute scores
imposter_scores = 1 / (1 + imposter_distances)
genuine_scores = 1 / (1 + genuine_distances)
# Print results
print('Imposter score: ', np.mean(imposter_scores))
print('Genuine score: ', np.mean(genuine_scores))


# In[235]:


svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_scaled, y_train)
y_pred_prob = svm.predict_proba(X_test_scaled)[:,1]
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
far = fp / (fp + tn)
frr = fn / (fn + tp)
eer = (far + frr) / 2
print('False Acceptance Rate (FAR): %0.2f' % far)
print('False Rejection Rate (FRR): %0.2f' % frr)
print('Equal Error Rate (EER): %0.2f' % eer)


# In[236]:


# Create a range of thresholds to test
thresholds = np.arange(0,1.1,0.1)

# Initialize FAR values
far_values = []
frr_values = []

# Loop through the thresholds and calculate the FAR for each
for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    far_values.append(far)
    frr_values.append(frr)
# Plot the results
plt.plot(thresholds, frr_values, '-o')
plt.xlabel('Threshold')
plt.ylabel('False Rejection Rate (FRR)')
plt.title('FRR vs. Threshold')
plt.show()


# In[237]:


X = data.iloc[:,1:32]
y = data.iloc[:, -1]
# Perform t-test for each feature
t_values = []
p_values = []
for col in X.columns:
    group1 = X[y == 0][col]
    group2 = X[y == 1][col]
    t, p = ttest_ind(group1, group2)
    t_values.append(t)
    p_values.append(p)
    # Print t-values and p-values for each feature
for col, t_val, p_val in zip(X.columns, t_values, p_values):
    print(f"{col}: t-value={t_val:.2f}, p-value={p_val:.4f}")
   # Create bar chart for t-values and p-values
fig, ax = plt.subplots()
x = np.arange(len(X.columns))
ax.bar(x - 0.2, t_values, width=0.6, label='t-value')
ax.bar(x + 0.2, p_values, width=0.6, label='p-value')
ax.set_xticks(x)
ax.set_xticklabels(X.columns, rotation=90)
ax.legend()
plt.show()


# In[238]:


from scipy.stats import ttest_ind
from sklearn.naive_bayes import GaussianNB

# Perform t-tests for each column of data
good_columns = []
bad_columns = []
p_value_threshold = 0.05

for col in data.columns[1:32]:
    t, p = ttest_ind(Imposter_data[col], Genuine_data[col])
    if p < p_value_threshold:
        good_columns.append(col)
    else:
        bad_columns.append(col)

# Create separate datasets for good and bad groups
good_data = data[good_columns]
bad_data = data[bad_columns]
print("*")
aa = [1,2,3]
bb = [1.1,2.1,3.1]

t, p = ttest_ind(aa,bb)
print(t)
print(p)
aa = [1,2,3]
bb = [1.5,2.5,3.5]
t, p = ttest_ind(bb,aa)
print(t)
print(p)

# Function to train classifiers, calculate FAR, and plot trade-off curves
def plot_tradeoff_curves(X_train, y_train, classifiers, labels, title):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for clf, label in zip(classifiers, labels):
        clf.fit(X_train_scaled, y_train)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        far = fpr * (1 - tpr)
        plt.plot(thresholds, far, label=label)

    plt.xlabel("Threshold")
    plt.ylabel("False Acceptance Rate (FAR)")
    plt.title(title)
    plt.legend()
    plt.show()

# Train classifiers, calculate FAR, and plot trade-off curves for good and bad groups
classifiers = [SVC(probability=True), GaussianNB(), DecisionTreeClassifier()]
labels = ["SVM", "Naive Bayes", "Decision Tree"]

plot_tradeoff_curves(good_data, y_train, classifiers, labels, "Good Group Keystroke Performance Trade-off Curves")
plot_tradeoff_curves(bad_data, y_train, classifiers, labels, "Bad Group Keystroke Performance Trade-off Curves")



# In[239]:


from scipy.spatial import distance
# load the data
df_train = pd.read_csv('finaldata.csv')
df_test = pd.read_csv('finaltest.csv')

# Select only the genuine data from the training dataset
df_train_genuine = df_train[df_train['Target'] == 1]

# Drop the target column to have only features
features_train_genuine = df_train_genuine.drop(columns=['Target'])
features_train_genuine = df_train_genuine.drop(columns=['user'])

# Create the genuine data template
template = features_train_genuine.agg(['mean', 'std'])

# Print the genuine data template
print("Genuine Data Template:")
print(template)

# Select only the genuine data from the testing dataset
df_test_genuine = df_test[df_test['Target'] == 1]
# Drop the target column to have only features
features_test_genuine = df_test_genuine.drop(columns=['Target'])
features_test_genuine = df_test_genuine.drop(columns=['user'])

# Calculate Euclidean and Manhattan distances from the template
euclidean_distances = features_test_genuine.apply(lambda row: distance.euclidean(row, template.loc['mean']), axis=1)
manhattan_distances = features_test_genuine.apply(lambda row: distance.cityblock(row, template.loc['mean']), axis=1)
# Print the distances
print("Euclidean Distances:")
print(euclidean_distances)
print("Manhattan Distances:")
print(manhattan_distances)

# # Plot the trade-off curve
# plt.figure(figsize=(10, 6))
# plt.plot(euclidean_distances, manhattan_distances)
# plt.title('Trade-off Curve between Euclidean and Manhattan Distances')
# plt.xlabel('Euclidean Distance')
# plt.ylabel('Manhattan Distance')
# plt.grid(True)
# plt.show()




# In[240]:


from scipy.spatial import distance
from numpy.linalg import pinv, det
from scipy.stats import multivariate_normal

# Calculate the pseudo-inverse covariance matrix and mean vector of the training data
cov = np.cov(features_train_genuine.values.T)
inv_cov = np.linalg.pinv(cov)  # pseudo-inverse
mean_vec = template.loc['mean'].values
print(mean_vec)
# Define a function to calculate Mahalanobis distance
def mahalanobis(x):
    diff = x - mean_vec
    return np.sqrt(diff.T.dot(inv_cov).dot(diff))

# Apply the function to each row in the testing data
mahalanobis_distances = features_test_genuine.apply(mahalanobis, axis=1)

# Print the Mahalanobis distances
print("Mahalanobis Distances:")
print(mahalanobis_distances)


# In[241]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Train a Manhattan Classifier (which is KNN with p=1)
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

# Predict the labels on your test data

y_pred = knn.predict(X_test)
# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate FAR and FRR
FAR = fp / (fp + tn)
FRR = fn / (fn + tp)

# Plot the tradeoff curve
plt.figure()
plt.plot([0, 1], [FAR, FRR], marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Genuine User')
plt.ylabel('FAR ')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[243]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]



# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FRRs = []

# Train a Manhattan Classifier (which is KNN with p=1)
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

# Generate different proportions of genuine and imposter data, calculate FRR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FRR and store it
    FRR = fn / (fn + tp)
    FRRs.append(FRR)
    
# Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FRRs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')

plt.ylabel('FRR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[244]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]

# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FARs = []


# Train a Manhattan Classifier (which is KNN with p=1)
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

# Generate different proportions of genuine and imposter data, calculate FAR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)
     # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FAR and store it
    FAR = fp / (fp + tn)
    FARs.append(FAR)

# Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FARs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')
plt.ylabel('FAR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[245]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]

# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FARs = []
# Train a KNeighborsClassifier with Euclidean distance (which is KNN with p=2)
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)

# Generate different proportions of genuine and imposter data, calculate FAR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)

    # Calculate the confusion matrix
    
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FAR and store it
    FAR = fp / (fp + tn)
    FARs.append(FAR)
    # Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FARs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')
plt.ylabel('FAR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[246]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]

# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FRRs = []



# Train a KNeighborsClassifier with Euclidean distance (which is KNN with p=2)
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)
# Generate different proportions of genuine and imposter data, calculate FRR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FRR and store it
    FRR = fn / (fn + tp)
    FRRs.append(FRR)
    
# Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FRRs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')
plt.ylabel('FRR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[247]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.pinv(X_train.cov().values)
# Define the Mahalanobis metric function
mahalanobis = distance.mahalanobis(u=X_train.mean(), v=X_test.mean(), VI=cov_inv)
print("Mahalanobis Distance: ", mahalanobis)

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]
# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FARs = []

# Train a KNeighborsClassifier with Mahalanobis distance
knn = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params={'VI': cov_inv})
knn.fit(X_train, y_train)

# Generate different proportions of genuine and imposter data, calculate FAR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FAR and store it
    FAR = fp / (fp + tn)
    FARs.append(FAR)
    # Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FARs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')
plt.ylabel('FAR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[248]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.pinv(X_train.cov().values)

# Define the Mahalanobis metric function
mahalanobis = distance.mahalanobis(u=X_train.mean(), v=X_test.mean(), VI=cov_inv)
print("Mahalanobis Distance: ", mahalanobis)

# Separate genuine and imposter data
genuine_test = X_test[y_test == 1]
imposter_test = X_test[y_test == 0]

genuine_labels = y_test[y_test == 1]
imposter_labels = y_test[y_test == 0]


# Prepare for storing results
proportions = np.linspace(0, 1, 50)  # Adjust this to change the number of points in the graph
FRRs = []

# Train a KNeighborsClassifier with Mahalanobis distance
knn = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params={'VI': cov_inv})
knn.fit(X_train, y_train)

# Generate different proportions of genuine and imposter data, calculate FRR for each
for p in proportions:
    num_genuine = int(p * genuine_test.shape[0])
    num_imposter = genuine_test.shape[0] - num_genuine

    X_subtest = pd.concat([genuine_test[:num_genuine], imposter_test[:num_imposter]])
    y_subtest = pd.concat([genuine_labels[:num_genuine], imposter_labels[:num_imposter]])

    y_pred = knn.predict(X_subtest)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_subtest, y_pred).ravel()

    # Calculate FRR and store it
    FRR = fn / (fn + tp)
    FRRs.append(FRR)
    # Plot the tradeoff curve
plt.figure()
plt.plot(proportions, FRRs, marker='o')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Proportion of Genuine Users')
plt.ylabel('FRR')
plt.title('Tradeoff Curve')
plt.grid(True)
plt.show()


# In[249]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, pairwise_distances
from scipy.spatial import distance
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.pinv(X_train.cov().values)
# Train a KNeighborsClassifier with Manhattan distance
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

# Compute scores based on the distance to the nearest neighbor
distances, _ = knn.kneighbors(X_test)
scores = 1.0 / (1.0 + distances)  # Convert distances to scores

# Separate genuine and imposter scores
genuine_scores = scores[y_test == 1]
imposter_scores = scores[y_test == 0]
print("Genuine Scores: ", genuine_scores)

# Plot the score distribution
sns.distplot(genuine_scores, hist=False, label='Genuine')
plt.xlim([0, 1])
plt.xlabel('Score')
plt.ylabel('Probability Density')
plt.title('Score Distribution')


plt.grid(True)
plt.legend()
plt.show()


# In[250]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.pinv(X_train.cov().values)

# Define distance metrics
metrics = ['manhattan', 'euclidean', 'mahalanobis']
metric_params = [None, None, {'VI': cov_inv}]

for metric, metric_param in zip(metrics, metric_params):
    # Train a KNeighborsClassifier with the given distance metric
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric, metric_params=metric_param)
    knn.fit(X_train, y_train)

    # Compute scores based on the distance to the nearest neighbor
    distances, _ = knn.kneighbors(X_test)
    scores = 1.0 / (1.0 + distances)  # Convert distances to scores
    
    # Separate genuine and imposter scores
    genuine_scores = scores[y_test == 1]

    print(f"Genuine Scores ({metric}): ", genuine_scores)

    # Plot the score distribution
    sns.distplot(genuine_scores, hist=False, label=metric)
    
plt.xlim([0, 1])
plt.xlabel('Score')
plt.ylabel('Probability Density')
plt.title('Score Distribution')
plt.grid(True)
plt.legend()
plt.show()


# In[251]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix
cov_inv = np.linalg.pinv(X_train.cov().values)

# Define distance metrics
metrics = ['manhattan', 'euclidean', 'mahalanobis']
metric_params = [None, None, {'VI': cov_inv}]



for metric, metric_param in zip(metrics, metric_params):
    # Train a KNeighborsClassifier with the given distance metric
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric, metric_params=metric_param)
    knn.fit(X_train, y_train)
 # Compute scores based on the distance to the nearest neighbor
    distances, _ = knn.kneighbors(X_test)
    scores = 1.0 / (1.0 + distances)  # Convert distances to scores

    # Separate genuine and imposter scores
    genuine_scores = scores[y_test == 1]

    # Calculate and print the mean and standard deviation of the genuine scores
    mean = np.mean(genuine_scores)
    std = np.std(genuine_scores)
    print(f"{metric} - Mean: {mean}, Std: {std}")

    # Plot the score distribution
    sns.distplot(genuine_scores, hist=False, label=metric)
    plt.xlim([0, 1])
plt.xlabel('Score')
plt.ylabel('Probability Density')
plt.title('Score Distribution')
plt.grid(True)
plt.legend()
plt.show()


# In[252]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Train a KNeighborsClassifier with Manhattan distance
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)
# Compute distances to the nearest neighbor
distances, _ = knn.kneighbors(X_test)

# Calculate thresholds as unique distances
thresholds = np.unique(distances)
print(thresholds)
# Prepare for storing results
error_rates = []
# Calculate error rate for each threshold
for thresh in thresholds:
    y_pred = (distances <= thresh).astype(int)
    error_rate = zero_one_loss(y_test, y_pred)
    error_rates.append(error_rate)

# Plot the trade-off curve
plt.figure()
plt.plot(thresholds, error_rates, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')

plt.title('Trade-off Curve')
plt.grid(True)
plt.show()


# In[253]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix for Mahalanobis distance
cov_inv = np.linalg.pinv(X_train.cov().values)
# Define distance metrics
metrics = ['manhattan', 'euclidean', 'mahalanobis']
metric_params = [None, None, {'VI': cov_inv}]

plt.figure()

for metric, metric_param in zip(metrics, metric_params):
    # Train a KNeighborsClassifier with the given distance metric
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric, metric_params=metric_param)
    knn.fit(X_train, y_train)
        # Compute distances to the nearest neighbor
    distances, _ = knn.kneighbors(X_test)

    # Calculate thresholds as unique distances and select the first 10
    thresholds = np.unique(distances)[:10]
    print(f"10 thresholds for {metric}: ", thresholds)

    # Prepare for storing results
    error_rates = []
# Calculate error rate for each threshold
    for thresh in thresholds:
        y_pred = (distances <= thresh).astype(int)
        error_rate = zero_one_loss(y_test, y_pred)
        error_rates.append(error_rate)

    # Plot the trade-off curve
    plt.plot(thresholds, error_rates, marker='o', label=metric)

plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('Trade-off Curve')
plt.grid(True)
plt.legend()
plt.show()


# In[256]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
import seaborn as sns

# Load the data
train = pd.read_csv('finaldata.csv')
test = pd.read_csv('finaltest.csv')

X_train = train.iloc[:, 1:32]
y_train = train['Target']

X_test = test.iloc[:, 1:32]
y_test = test['Target']

# Calculate the inverse of the covariance matrix for Mahalanobis distance
cov_inv = np.linalg.pinv(X_train.cov().values)

# Define distance metrics
metrics = ['manhattan', 'euclidean', 'mahalanobis']
metric_params = [None, None, {'VI': cov_inv}]

for metric, metric_param in zip(metrics, metric_params):
    # Train a KNeighborsClassifier with the given distance metric
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric, metric_params=metric_param)
    knn.fit(X_train, y_train)

    # Compute distances to the nearest neighbor
    distances, _ = knn.kneighbors(X_test)
    

    # Compute FAR and FRR
    fpr, tpr, thresholds = roc_curve(y_test, distances[:, 0], pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    # Plot EER
    plt.figure()
    plt.title(f'FAR-FRR Trade-off using {metric} Distance')
    plt.plot(thresholds, fpr, color='blue', lw=1, label='FAR')
plt.plot(thresholds, fnr, color='red', lw=1, label='FRR')
plt.xlim([0.0, max(thresholds)])
plt.ylim([0.0, 1.0])
plt.plot([eer_threshold], [fpr[np.nanargmin(np.absolute((fnr - fpr)))]], marker='o', markersize=5, color="green", label='EER')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


# In[257]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('finaldata.csv')
X = data.iloc[:, 1:32]
y = data['Target']

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    
    "Logistic Regression": LogisticRegression(),
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

plt.figure()
# Iterate over classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=f"{name}, AUC={auc:.3f}")
    
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Plot line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curves')

plt.legend(loc="lower right")
plt.show()


# In[258]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the data
data = pd.read_csv('finaldata.csv')
X = data.iloc[:, 1:32]
y = data['Target']

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Classifier": make_pipeline(StandardScaler(), SVC(probability=True)), # SVC needs scaling
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}
plt.figure()

# Iterate over classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=f"{name}, AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Plot line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curves')
plt.legend(loc="lower right")
plt.show()


# In[259]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the data
data = pd.read_csv('finaldata.csv')
X = data.iloc[:, 1:32]
y = data['Target']
# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Classifier": make_pipeline(StandardScaler(), SVC(probability=True)), # SVC needs scaling
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to store AUCs
aucs = {}

plt.figure()
# Iterate over classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    aucs[name] = auc
    plt.plot(fpr,tpr,label=f"{name}, AUC={auc:.3f}")
    
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Plot line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curves')
plt.legend(loc="lower right")
plt.show()


# In[260]:


# Identify the best classifier
best_classifier = max(aucs, key=aucs.get)
print(f"The best classifier is: {best_classifier} with AUC: {aucs[best_classifier]:.3f}")


# In[ ]:

# def check_pattern(pattern):
def check_pattern(pattern):
    ar = np.array(pattern, dtype=float)
    arr = ar.reshape(1, -1)
    arr = scaler.transform(arr)
    res = classifier.predict(arr)
    return "Genuine" if res[0] == 1 else "Imposter"


