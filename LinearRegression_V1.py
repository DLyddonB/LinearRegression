import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Load the csv file
data = pd.read_csv('1.01.+Simple+linear+regression.csv')

#Verify that I have actually loaded in the data set (show me the first 10 entries?)
print(data.head(10))
#print(data.describe())

y = data['GPA']
x1 = data['SAT']

#Each point in the graph represents a student
plt.scatter(x1,y)

#So we got these numbers by looking in the sm.OLS table (coef const = 0.2750, coef SAT = 0.0017)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, color='orange', label='Regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())



