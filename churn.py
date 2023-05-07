#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


X = pd.read_csv('your-file-path.csv')

y = X.churn_probability
# y represents the dependent variable we wanted to predict  
# in this case we are wanting to predict churn probabilities of customers from some telecom company          
 
X.drop(['churn_probability'], axis=1, inplace=True)


# IF YOU HAVE COLUMNS THAT HAVE HIGH MISSING VALUES, YOU CAN DELETE THEM BY USING FOLLOWING CODE

# In[ ]:


limit = len(X) * .80 #you can adjust the limit by chancing allowed percentage of missing values
X = X.dropna(thresh=limit, axis=1)


# In[ ]:


# getting datetime columns
date_cols = list(X.select_dtypes(include=['object', 'category']).columns)


# adjust the threshold for low_cardinality columns as needed to capture all of your categorical variables.
low_cardinality_cols = [col for col in X.columns if X[col].nunique() < 3]

# categorical cols also includes datetime values, remove them
cat_cols = [col for col in  low_cardinality_cols if col not in date_cols]

num_cols = [col for col in X.columns if col not in date_cols + cat_cols]

# not do any preprocessing in id, because we predict each customer's churn prob
num_cols.remove('id')


# you can make adjustments for checking if you capture all the columns
#print('All Columns:', X.columns)
#print('Numerical Columns:', num_cols)
#print('Categorical Columns:',cat_cols)
#print('Date Columns:',(date_cols)


# In[ ]:


# parameters can be changed for obtaining more accurate predictions
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())])


cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

date_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ('date', date_transformer, date_cols)])


# full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs', max_iter=10000))])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf.fit(X_train, y_train)


# In[ ]:


pred = clf.predict(X_test)


# In[ ]:


print("model score: %.3f" % clf.score(X_test, y_test))


# In[ ]:


# creating new dataframe which contains customer ids and churn probabilities

pred = pd.DataFrame(pred,columns=['churn_probability'])
ID = pd.DataFrame(X_test.index,columns=['id'])
sub = pd.concat([ID,pred],axis=1)
sub.set_index('id',inplace=True)


# converting data frame into csv file
sub.to_csv(f"submission.csv")

