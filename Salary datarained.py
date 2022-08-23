#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('Salary.csv')


# In[35]:


df


# In[36]:


df.head()


# In[37]:


df.tail()


# In[38]:


df.columns


# In[39]:


df.isnull().sum()


# In[40]:


import seaborn as sns


# In[41]:


sns.scatterplot(x="rank",y="salary",data=df)


# In[42]:


sns.scatterplot(x="discipline",y="salary",data=df)


# In[43]:


sns.scatterplot(x="yrs.since.phd",y="salary",data=df)


# In[44]:


sns.scatterplot(x="yrs.service",y="salary",data=df)


# In[45]:


sns.scatterplot(x="sex",y="salary",data=df)


# In[46]:


import matplotlib.pyplot as plt
sns.pairplot(df)
plt.savefig('pairplot.png')
plt.show()


# In[47]:


df.corr()


# In[48]:


df.corr()['salary'].sort_values()


# In[49]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,linecolor="black",fmt='.2f')


# In[50]:


df.describe()


# In[51]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
sns.heatmap(round(df.describe()[1:].transpose(),2),linewidth=2,annot=False,fmt='f')
plt.xticks(fontsize=18)
plt.yticks(fontsize=12)
plt.title("Variable summary")
plt.savefig('heatmaap.png')
plt.show()


# In[52]:


df.info()


# In[53]:


df.skew()


# In[54]:


sns.distplot(df["yrs.since.phd"])


# In[55]:


sns.distplot(df["yrs.service"])


# In[ ]:


sns.distplot(df["salary"])


# In[ ]:


df.corr()['salary']


# In[56]:




delete=pd.DataFrame([["0.334745","yrs.service","No","Alot"]],columns=["correlation with Target","Column NAme","Normalized","Outliers"])
delete


# In[57]:


df=df.drop(["yrs.service"],axis=1)


# In[ ]:


df


# In[58]:


df.salary.value_counts()


# In[59]:


df.salary.unique()


# In[60]:


df.dtypes


# In[61]:


from scipy.stats import zscore


# In[62]:


import numpy as np


# In[63]:


np.abs(zscore(df))


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import accuracy_score


# In[ ]:


lr=LogisticRegression()
for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=i,test_size=0.20)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_train=lr.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,1)==round(accuracy_score(y_test,pred_test)*100,1):
        print("At random state",i,"The model perform very well")
        print("At random_state:-",i)
        print("Training accuracy score is:-",round(accuracy_score(y_train,pred_train)*100,1))
        print("Testing accuracy_score is:-",round(accuracy_score(y_test,pred_test)*100,1),'\n\n')


# In[ ]:




