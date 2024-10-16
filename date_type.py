import pandas as pd
import numpy as np
import matplotlib as mp


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)

na_vals = ['Missing','']
d_parser = lambda x : pd.datetime.strptime(x, '%Y-%m-%d %I-%p')
df = pd.read_csv('D:\\Python\\Pandas\\Prices\\ETH_1h.csv', parse_dates=['Date'], date_parser=d_parser)#parsing dates on loading
#schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)
print(df.loc[0, 'Date'])
#df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %I-%p')# parsing date when loading csv
print(df.loc[0, 'Date'].day_name())
df['DayOfWeek']= df['Date'].dt.day_name()

df['Date'].min()

dt_filter = (df['Date'] >= '2020') & (df['Date'] <= '2021')
dt_filter = (df['Date'] >=pd.to_datetime('2020-01-01')) & (df['Date'] <= pd.to_datetime('2021-01-01'))


print(df.loc[dt_filter])

#using date slicing

df.set_index('Date', inplace=True)

print(df['2020-01':'2021-01'])

#Average closing price
print(df['2020-01':'2021-01']['Close'].mean())

#getting highest trades for each day

#resample data to days and selex max for each day
highs = df['High'].resample('D').max()
print(highs.plot())