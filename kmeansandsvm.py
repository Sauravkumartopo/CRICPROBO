#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import warnings
#from google.colab import drive
from mpl_toolkits import mplot3d
from pylab import rcParams
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
     

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
pd.set_option('display.expand_frame_repr', False)
rcParams['figure.figsize'] = 14, 7
     


     

df = pd.read_csv("C:/Users/saura/Downloads/archive/train.csv")
     

df.head()


# In[3]:


print ("Total number of rows in dataset = {}".format(df.shape[0]))
print ("Total number of columns in dataset = {}".format(df.shape[1]))


# In[4]:


j = sns.jointplot("X1", "X2", data = df, kind = 'reg')
j.annotate(stats.pearsonr)
plt.show()


# In[7]:


target_col = "price_range"
X = df.loc[:, df.columns != target_col]
y = df.loc[:, target_col]


# In[8]:


k = 3 #number of variables for heatmap
cols = df.corr().nlargest(k, target_col)[target_col].index
cm = df[cols].corr()
plt.figure(figsize=(14,8))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# In[9]:


X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant)


# In[10]:


results = model.fit()
print(results.summary())


# In[11]:


lasso = Lasso()
params = {"alpha" : [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 1e1, 
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}


# In[12]:


lasso_regressor = GridSearchCV(lasso, params, 
                               scoring="neg_mean_squared_error", 
                               cv=5)


# In[13]:


lasso_regressor.fit(X, y)


# In[14]:


lasso_regressor.best_score_


# In[15]:


lasso_regressor.best_estimator_


# In[16]:


lasso_best = lasso_regressor.best_estimator_


# In[17]:


lasso_best.fit(X, y)


# In[18]:


coef = pd.Series(lasso_best.coef_,list(X.columns))
coef.plot(kind='bar', title='Model Coefficients')


# In[ ]:




