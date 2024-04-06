import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()
import unicodeit

data = pd.read_csv('real_estate_price_size.csv')
print(data.describe())

y = data['price']
x1 = data['size']

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

plt.scatter(x1,y)
yhat = 223.1787*x1 + 101900
fig = plt.plot(x1, yhat, lw=2, color='red', linestyle='dotted', label='Regression line')
plt.xlabel(unicodeit.replace('Size (m^2)'),fontsize = 15)
plt.ylabel('Price',fontsize = 15)

#plt.savefig('LinearRegression_PropertyPriceSize.png', dpi=300, bbox_inches='tight')
plt.show()