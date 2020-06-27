import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

covid_data = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
covid_data = covid_data.drop(
    ['FIPS', 'Admin2', 'Lat', 'Long_', 'iso2', 'iso3', 'UID', 'code3', 'Country_Region', 'Combined_Key'], axis=1)
covid_data = covid_data.groupby('Province_State', axis=0).sum()
covid_data = covid_data.transpose()
covid_data.index = pd.to_datetime(covid_data.index)
covid_data['Total'] = covid_data.sum(axis=1)
covid_data['Daily'] = covid_data['Total'].diff()

model = ARIMA(covid_data[['Daily']], order=(5, 1, 0))
model_fit = model.fit()
with open('model-summary.txt', 'w') as f:
    for line in model_fit.summary().tables:
        f.write(line.as_text())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.savefig('arima-residuals.png', format='png')
plt.show()
residuals.plot(kind='kde')
plt.savefig('arima-residuals-kde.png', format='png')
plt.show()
with open('residual-description.txt', 'w') as f:
    f.write(residuals.describe().to_string())
