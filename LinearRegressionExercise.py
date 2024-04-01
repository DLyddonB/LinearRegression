import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#load the csv file using the pandas method
data = pd.read_csv('real_estate_price_size.csv')
#print the first 10 entries
print(data.head(10))
#Pandas method for giving useful statistics (observations [count], mean, st.dev etc)
data.describe()

y = data['price']
x1 = data['size']

#plot scatter graph
plt.scatter(x1, y)
yhat = 223.1787*x1 + 101900
# this is the regression line which is to say that it's the best fitting line, or the line which is
# closest to all observations simultaneously
fig = plt.plot(x1,yhat, lw=4, color='orange', label='Regression line')
plt.xlabel('Size (m^2)', fontsize=15)
plt.ylabel('Price ($)', fontsize=15)
plt.show()

x = sm.add_constant(x1)
#Results containts the output of the Ordinary Least Squares (OLS) regression. As arguments we
# need to add the dependent variable (y) and the newly defined x. fit() will apply a specific estimation technique (OLS in this case) to obtain
# the fit of the model
results = sm.OLS(y,x).fit()
print(results.summary())

#adding in another comment to test out the versioning