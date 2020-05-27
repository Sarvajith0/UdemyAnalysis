# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:13:25 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv(r"C:\Users\Andrew\Documents\Python Scripts\data set\udemy_courses.csv")

df.info()
df.head()

df["published_timestamp"] = pd.to_datetime(df["published_timestamp"])

'''exploring data distributions'''
num_cols_df = df.select_dtypes(['int64','float64','datetime64[ns, UTC]'])
cat_cols_df = df.select_dtypes(['object'])

num_cols_df.hist(figsize = (10,10),bins=100)

(cat_cols_df['level'].value_counts()/cat_cols_df.shape[0]).plot(kind="bar")
(cat_cols_df['subject'].value_counts()/cat_cols_df.shape[0]).plot(kind="bar")

(cat_cols_df[cat_cols_df['subject']=='Business Finance']['level'].value_counts()/cat_cols_df.shape[0]).plot(kind="bar")

'''creating new features'''
num_cols_df["average_lecture_length"] = num_cols_df["content_duration"]/num_cols_df["num_lectures"]
num_cols_df["average_lecture_length"].hist(figsize=(10,10), bins = 100)
num_cols_df["average_lecture_length"].mean()

num_cols_df["revenue"] = num_cols_df["price"]*num_cols_df["num_subscribers"]
num_cols_df["revenue"].hist(figsize=(10,10), bins = 100)
num_cols_df["revenue"].mean()

'''comparing features'''
temp_df = pd.concat([num_cols_df,cat_cols_df], axis = 1)
sns.pairplot(temp_df, x_vars = ['content_duration','num_lectures','num_reviews','average_lecture_length'],y_vars = ['num_subscribers','price','revenue'], hue = 'level')

'''preprocessing'''
cat_cols_df = cat_cols_df.iloc[:,2:]

def create_dummy_df(df, cat_cols, dummy_na):
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df

cat_cols_df = create_dummy_df(cat_cols_df, cat_cols_df.columns, dummy_na = False)

from sklearn.model_selection import train_test_split

X_num_cols_df = num_cols_df[['content_duration','num_lectures','num_reviews','price']]

X = pd.concat([X_num_cols_df,cat_cols_df],axis = 1)
y = num_cols_df['revenue']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

'''running the model'''
from sklearn.linear_model import LinearRegression
lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit

from sklearn.metrics import mean_squared_error
y_test_preds = lm_model.predict(X_test)
mse_score = mean_squared_error(y_test, y_test_preds)
length_y_test = len(y_test)
print("The MSE for your model was {} on {} values.".format(mse_score, length_y_test))

lm_model.coef_