import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

start_date = dt.datetime(2020, 1, 22)
end_date = dt.datetime(2020, 6, 20)

covid_data = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
covid_data = covid_data.drop(
    ['FIPS', 'Admin2', 'Lat', 'Long_', 'iso2', 'iso3', 'UID', 'code3', 'Country_Region', 'Combined_Key'], axis=1)
covid_data = covid_data.groupby('Province_State', axis=0).sum()
covid_data = covid_data.transpose()
covid_data.index = pd.to_datetime(covid_data.index)
covid_data['Total'] = covid_data.sum(axis=1)
covid_data['Daily'] = covid_data['Total'].diff()

model = covid_data[['Daily']]
model['SMA7'] = model['Daily'].rolling(7).mean()
model['SMA30'] = model['Daily'].rolling(30).mean()
# model.dropna(inplace=True)
model['Positions'] = np.where(model['SMA7'] > model['SMA30'], 1.0, 0.0)
model['Crossover'] = model['Positions'].diff()

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='US Daily Cases')
ax1.plot(model.loc[model['Crossover'] == 1.0].index,
         model['SMA7'].loc[model['Crossover'] == 1.0], '^', markersize=10, zorder=5, label='SMA7 > SMA30')
ax1.plot(model.loc[model['Crossover'] == -1.0].index,
         model['SMA7'].loc[model['Crossover'] == -1.0], 'v', markersize=10, zorder=4, label='SMA7 < SMA30')
model['Daily'].plot(ax=ax1, lw=2., zorder=1)
model['SMA7'].plot(ax=ax1, lw=2., zorder=3)
model['SMA30'].plot(ax=ax1, lw=2., zorder=2)
fig.add_axes(ax1)
plt.legend()
plt.savefig('covid-crossover.png', format='png')
plt.show()
