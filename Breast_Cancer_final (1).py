#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os 
os.chdir(r"C:\Users\venuk\OneDrive\Desktop\imarticus")
os.getcwd()


# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:


my_data=pd.read_csv("cancerdata.csv")


# In[50]:


my_data["diagnosis"]=my_data["diagnosis"].map({"M":1,"B":0})


# In[51]:


my_data=my_data.drop(['id'],1)


# In[52]:


my_data["diagnosis"].value_counts()      #62 38   


# In[53]:


from sklearn.utils import resample
df_majority=my_data[my_data.diagnosis==0]
df_minority=my_data[my_data.diagnosis==1]


# In[54]:


df_minority_upsampled=resample(df_minority,replace=True,n_samples=357,random_state=123)
my_data_upsampled=pd.concat([df_majority,df_minority_upsampled])


# In[55]:


my_data_upsampled["diagnosis"].value_counts() 


# In[56]:


my_data_upsampled=my_data_upsampled.iloc[:,0:11]


# In[57]:


my_data_upsampled.shape


# In[58]:


my_data_upsampled=my_data_upsampled.drop(["perimeter_mean"],1)
my_data_upsampled.columns


# In[59]:


my_data_upsampled.columns


# In[60]:


X=my_data_upsampled.drop(['diagnosis'],1)
y=my_data_upsampled['diagnosis']


# In[61]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state= 123)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[63]:


y_pred = knn.predict(X_test)


# In[64]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[65]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[66]:


import pickle
filename = 'breast_cancer_model.pkl'
pickle.dump(knn, open(filename, 'wb'))


# In[67]:


model = pickle.load(open(filename, 'rb'))
output=model.predict([[20.84,22.11,273.9,0.103,0.11,0.094,0.045,0.17,0.06]])


# In[68]:


output


# In[ ]:




