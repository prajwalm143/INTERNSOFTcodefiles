# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:10:30 2026

@author: ADMIN
"""

#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt



#READING THE DATA FROM YOUR FILES
data = pd.read_csv("advertising.csv")
data.head()




#TO VISUALISE DATA
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='radio',y='sales',ax=axs[1])
data.plot(kind='scatter',x='newspaper',y='sales',ax=axs[2])




#CREATING X&Y FOR LINEAR REGRESSION
feature_cols = ['TV']
x = data[feature_cols]
y = data.sales

#IMPORTING LINEAR REGRESSION ALGO FOR SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


#CREATE A DATAFRAM WITH AND MAX VALUE OF THE TABLE
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
preds

data.plot(kind='scatter', x='TV', y='sales')

plt.plot(X_new,preds,c='red',linewidth=3)




import statsmodels.formula.api as smf
lm = smf.ols(formula='sales ~ TV', data=data).fit()
lm.conf_int()

#FINDING THE PROBABLITY VALUES
lm.pvalues


#FINDING THE R-SQUARED VALUES
lm.rsquared

#MULITI LINEAR REGRESSION
feature_cols = ['TV','radio','newspaper']
X = data[feature_cols]
y = data.sales

lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula='sales ~ TV+radio+newspaper',data=data).fit()
lm.conf_int()
lm.summary()

lm = smf.ols(formula='sales ~ TV+radio',data=data).fit()
lm.conf_int()
lm.summary()