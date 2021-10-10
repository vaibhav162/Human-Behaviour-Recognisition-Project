#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


train=pd.read_csv(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Human behaviour project\train.csv")
test=pd.read_csv(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Human behaviour project\test.csv")


# In[3]:


shuffle(train)
shuffle(test)


# In[4]:


train.head(2)


# In[5]:


train.tail(2)


# In[6]:


train.shape


# In[7]:


test.head(2)


# In[8]:


test.tail(2)


# In[9]:


test.shape


# In[10]:


# Checking for Null values

print("Any missing value in Training set", train.isnull().values.any())
print("Any missing value in Testing set", test.isnull().values.any())


# # Exploring Dataset

# In[11]:


train_outcome= pd.crosstab(index= train["Activity"], columns= "Count")
train_outcome


# # Exploratory Data Analysis

# In[12]:


temp= train["Activity"].value_counts()
temp


# In[13]:


df= pd.DataFrame({"labels": temp.index, "values": temp.values})


# In[14]:


df.head(2)


# In[15]:


labels= df["labels"]
sizes= df["values"]
colors= ["yellowgreen", "lightskyblue", "gold", "lightpink", "cyan", "lightcoral"]
patches, texts= plt.pie(sizes, colors=colors, labels=labels, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="right")
plt.axis("equal")
plt.tight_layout()
plt.show()


# # Data Processing

# In[16]:


X_train= pd.DataFrame(train.drop(["Activity", "subject"], axis=1))
Y_train_label= train.Activity.values.astype(object)
X_test= pd.DataFrame(test.drop(["Activity", "subject"], axis=1))
Y_test_label= test.Activity.values.astype(object)


# In[17]:


from sklearn import preprocessing
encoder= preprocessing.LabelEncoder()

encoder.fit(Y_train_label)
y_train= encoder.transform(Y_train_label)
y_train


# In[18]:


encoder.fit(Y_test_label)
y_test= encoder.transform(Y_test_label)
y_test


# In[19]:


num_cols= X_train._get_numeric_data().columns
num_cols


# In[20]:


num_cols.size


# # Model Building for Human Activity Recognition

# In[21]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train= scaler.fit_transform(X_train)
x_test= scaler.fit_transform(X_test)


# In[22]:


knn= KNeighborsClassifier(n_neighbors=24)
knn.fit(x_train, y_train)


# In[23]:


y_pred= knn.predict(x_test)


# In[26]:


print((accuracy_score(y_test, y_pred)*100), "%")


# In[24]:


scores=[]
for i in range(1,50):
    knn= KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train, y_train)
    y_pred= knn.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))


# In[25]:


plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy Score")
xticks= range(1,50)
plt.plot(xticks, scores, color="red", linestyle="solid", marker="o",
        markersize=5, markerfacecolor="blue")
plt.show()


# In[28]:


scores= np.array(scores)
print("Optimal number of neighbor is:", scores.argmax())
print("Accuracy Score:" +str(scores.max()*100), "%")


# # Conclusion

# In[29]:


knn= KNeighborsClassifier(n_neighbors=19)
knn.fit(x_train, y_train)
y_pred= knn.predict(x_test)
y_pred_label= list(encoder.inverse_transform(y_pred))
y_pred_label


# In[31]:


print(confusion_matrix(Y_test_label, y_pred_label))


# In[35]:


print(classification_report(Y_test_label, y_pred_label))


# In[ ]:




