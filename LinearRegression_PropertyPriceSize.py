import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()
import unicodeit

data = pd.read_csv('real_estate_price_size.csv')
#using the data.describe() method gives us nice descriptive statistics!
print(data.describe())

#y is the dependent variable
y = data['price']
#x is the independent variable
x1 = data['size']

#add a constant
x = sm.add_constant(x1)
#fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and independent x
results = sm.OLS(y, x).fit()
print(results.summary())

plt.scatter(x1,y)
#coefficients obtained from the OLS summary. In this case, the property size has the multiplier of 223.1787 and the intercept of 101900 (the constant)
yhat = 223.1787*x1 + 101900
fig = plt.plot(x1, yhat, lw=2, color='red', linestyle='dotted', label='Regression line')
plt.xlabel(unicodeit.replace('Size (m^2)'),fontsize = 15)
plt.ylabel('Price',fontsize = 15)

plt.show()
