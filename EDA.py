#!/usr/bin/env python
# coding: utf-8

# ![QuantConnect Logo](https://cdn.quantconnect.com/web/i/logo-small.png)
# 

# ## BT2101 Final Project

# ### I. Import Libraries

# In[268]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Imports
from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Jupyter")
AddReference("QuantConnect.Indicators")
from System import *
from QuantConnect import *
from QuantConnect.Data.Custom import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from QuantConnect.Jupyter import *
from QuantConnect.Indicators import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from  math import *

# Create an instance
qb = QuantBook()


# ### II. Adding Currency Pairs Data

# In[269]:


GBPJPY = qb.AddForex("GBPJPY") #chosen as base currency based on volatility
EURNZD = qb.AddForex("EURNZD")
GBPAUD = qb.AddForex("GBPAUD")
EURUSD = qb.AddForex("EURUSD")
USDJPY = qb.AddForex("USDJPY")
GBPUSD = qb.AddForex("GBPUSD")


# ### III. Exploratory Data Analysis (EDA)
# #### This section analyses and justifies the various currency pairs chosen. Our group compared the volatility of the currency pairs using Bollinger Bands. 

# #### Volatility Analysis of Chosen Currency (EURJPY)

# In[270]:


## modified BollingerBand function ##
def BollingerBand(cp, data, mean, std):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.set_title(cp)
    ax.set_xlabel('Time')
    ax.set_ylabel('Exchange Rate')
    
    x_axis = list(data.loc[data['symbol'] == cp]['time'])
    # x_axis = data.loc[data['symbol'] == cp].index.get_level_values(0)
    
    
    mean = data.loc[data['symbol'] == cp]['close']
    bollinger_upper = (mean + std*2)
    bollinger_lower = (mean - std*2)
    
    ax.fill_between(x_axis, bollinger_upper, bollinger_lower, facecolor='grey', alpha = 0.5)
    
    ax.plot(x_axis, mean, color='blue', lw=2)
    ax.plot(x_axis, bollinger_upper, color='green', lw=2)
    ax.plot(x_axis, bollinger_lower, color='orange', lw=2)
    
    ax.legend()
    plt.show();
    
    bollinger_upper = list(mean + std*2)
    bollinger_lower = list(mean - std*2)
    


# In[271]:


## modified function for plotting BollingerBand ##
'''
Based off some research done online, our group has decided to drill down further
into the following 3 currency pairs due to their relatively higher volatility.
The pair with the highest volatility was eventually chosen as the base pair acting
as the basis for our portfolio.
'''

volatileCP = ['GBPJPY', 'EURNZD', 'GBPAUD']
chosenCP = [0,0];
for cp in volatileCP:
    qb.AddForex(cp)
    h1 = qb.History(qb.Securities.Keys, 180, Resolution.Hour)
    h1_df = pd.DataFrame(h1)
    
    # adding the new variable 'mid' into the dataframe (justification found in report)
    h1_df['mid'] = (h1_df['high']+h1_df['low'])/2
    h1 = h1_df
    
    # convert row name 'time' from an index to a column 
    h1_df.index.name = 'time'
    h1_df.reset_index(inplace=True)
    get_data = h1_df.loc[h1_df['symbol'] == cp]['mid']
    data = h1_df
    
    t=10 # Rolling time window, and calculate mean and standard error
    rolmean = get_data.rolling(window=t).mean()
    rolstd = get_data.rolling(window=t).std()
    
    BollingerBand(cp, data, rolmean, rolstd)
    if(chosenCP[0]<get_data.describe()['std']):
        chosenCP[0] = get_data.describe()['std']
        chosenCP[1] = cp


# Based on volatility, as measured by std, we chose our base currency pair: GBPJPY

# In[272]:


chosenCP


# In[274]:


h1_df.loc[h1_df['symbol'] == 'GBPJPY']['mid']


# In[275]:


# get number of rows, columns
data_size = h1.shape[0:2]
data_size


# In[276]:


# Split the dataset: First 70% for training, Last 30% for testing
data_size = h1.shape[0]


data_train = h1_df.loc[h1_df['symbol'] == 'GBPJPY']['mid'].iloc[0:int(data_size*0.7)]
data_test = h1_df.loc[h1_df['symbol'] == 'GBPJPY']['mid'].iloc[int(data_size*0.7):]


# In[277]:


#plotting the time series for the 'mid' feature obtained from feature engineering
h1_df.loc[h1_df['symbol'] == 'GBPJPY']['mid'].plot()


# ### Rolling Time Window (t=3)

# In[279]:


t=3 # Rolling time window, and calculate mean and standard error
rolmean = data_train.rolling(window=t).mean()
rolstd = data_train.rolling(window=t).std()
rolmean, rolstd


# In[280]:


#h1_df.loc['GBPJPY']['mid'].plot()
rolmean.plot()
rolstd.plot()


# ### Dickey-Fuller Test on stationary condition
# #### This section checks if the data is stationary and removes trend & seasonality accordingly to do analysis on the residuals/ noise.

# In[281]:


# Dickey-Fuller test on stationary condition
from statsmodels.tsa.stattools import adfuller
print ('Results of Dickey-Fuller Test:')

# Dickey-Fuller test
dftest = adfuller(data_train, autolag=None) 
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)


# #### Dickey-Fuller test shows that null hypothesis (nonstationarity) cannot be rejected, so the time series may be nonstationary. If non-stationary, we cannot use AR(I)MA to fit the data, make inference or do forecasting. Thus, we need to make the time series stationary first. 
# 
# #### We have several ways to make time series stationary. 

# In[282]:


# Remove trend from original time series: Subtract this estimated trend
data_log_moving_avg_diff = data_train - rolmean
data_log_moving_avg_diff


# In[283]:


# Drop missing values
data_log_moving_avg_diff.dropna(inplace=True)
data_log_moving_avg_diff


# In[284]:


# Evaluating Trendless Time Series for Stationary: Using Dickey-Fuller test
print ('Results of Dickey-Fuller Test:')

# Dickey-Fuller test
dftest = adfuller(data_log_moving_avg_diff, autolag=None) 
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)


# #### This time, we find that Dickey-Fuller test is significant (p-value<5%). This means, after removing general trend, the time series data probably become stationary.
# 
# #### However, let us decompose the time series into trend, seasonality and residual to better understand the data

# ### Decomposition of Original Time Series
# #### into Trend, Seasonality and Residual/ Noise

# In[288]:


# Decomposing the Original Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(list(data_train), freq=15)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the components
plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(data_train, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# #### From the above, we can make a few conclusions.
# #### 1. There is a generally decreasing trend.
# #### 2. There is seasonality, on a very small scale.
# #### 3. The residuals seems to play a significant role in the currency price.

# ### Removing Seasonality
# #### Even though seasonality is observed on a very small scale, our group has decided to remove it.

# In[289]:


# Evaluating the Residuals for Stationary
data_decompose = pd.Series(residual)
data_decompose.dropna(inplace=True)

t=30 # Rolling time window, and calculate mean and standard error
rolmean = data_decompose.rolling(window=t).mean()
rolstd = data_decompose.rolling(window=t).std()

# Visualize whether mean and standard error of time series are stationary over time
plt.figure(figsize=(15,5))
orig = plt.plot(data_decompose, color='blue',label='Original Residuals')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation of Residuals')
plt.show(block=False)

# Dicky-Fuller test on stationary condition
print ('Results of Dickey-Fuller Test:')

# Dickey-Fuller test
dftest = adfuller(data_decompose, autolag=None) 
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)


# ### ARIMA

# In[290]:


from statsmodels.tsa.arima_model import ARIMA


# In[292]:


# Suppose we use ARIMA(p,d,q) to fit the time series data: AR(p), MA(q) and d-order-differencing
# Remember: we have tested that first-order-differencing is effective enough to make non-stationary to stationary 
# Here we can use ARIMA model, which fits ARMA model on (first-order) differencing time series data

from statsmodels.tsa.arima_model import ARIMA
#import seaborn as sns

# ARIMA(5, 1, 5): AR(5), MA(5) and first-order-differencing
model = ARIMA(data_train, order=(5,1,5))
model_fit = model.fit(disp=0)
print (model_fit.summary())
# Find AIC value
print ("AIC is: %s" %(model_fit.aic))

# Plot residual errors series
plt.figure()
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Plot distribution of residual errors
#sns.distplot(residuals, hist=True, kde=True, bins=50, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})


# In[294]:


# We can do a grid search on selecting autoregressive order or moving average order in ARIMA model
# Select the model with best performance
# Remember usually you need to further split training data to do validation, and find the model with best validation accuracy. 
# In this example, in order to simply and quickly illustrate the procedure of time series modelling, we ignore this step.
p_list = [i for i in range(1, 11)] # AR order
best_AIC = float('inf') # You need to minimize AIC
best_model = None # Model with lowest AIC

for p in p_list:
    # ARIMA(p, 1, 1): AR(p), MA(1) and first-order-differencing
    model = ARIMA(data_train, order=(p,1,1))
    model_fit = model.fit(disp=0)
        
    if model_fit.aic <= best_AIC:
        best_model, best_AIC = model, model_fit.aic

