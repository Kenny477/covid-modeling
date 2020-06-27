import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

covid_data = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
covid_data = covid_data.drop(
    ['FIPS', 'Admin2', 'Lat', 'Long_', 'iso2', 'iso3', 'UID', 'code3', 'Country_Region', 'Combined_Key'], axis=1)
covid_data = covid_data.groupby('Province_State', axis=0).sum()
covid_data = covid_data.transpose()
covid_data.index = pd.to_datetime(covid_data.index)
covid_data['Total'] = covid_data.sum(axis=1)
covid_data['Daily'] = covid_data['Total'].diff()

print('Autocorrelation: ' + str(covid_data['Daily'].autocorr(1)))
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Autocorrelation')
pd.plotting.autocorrelation_plot(covid_data[['Daily']].dropna(), ax=ax1)
plt.savefig('covid-autocorrelation.png', format='png')
plt.show()
