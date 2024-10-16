import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)

na_vals = ['Missing','']
df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv',  index_col= 'Respondent', na_values=na_vals)#creating index on loading
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)

#df.isna()

#df.dropna()# drops all rows with Nan value
#df.dropna(how='all')# drops all rows with all Nan values
#df.dropna(how='all', subset=['email'], inplace=True)# drops all rows with Nan email values

#replace custom values
#df.replace('NA', np.nan, inplace=True)

#Fill Nan Values
#df.fillna("Missing")#entire dataframe
#check datatypes
#df.dtypes

#convert datatype
#df['age'] = df['age'].astype(float)

#print(df['YearsCode'].value_counts())
print(df['YearsCode'].unique())
df['YearsCode'].replace('Less than 1 year', 0, inplace=True)
df['YearsCode'].replace('More than 50 years', 51, inplace=True)

df['YearsCode'] = df['YearsCode'].astype(float)
print(df['YearsCode'].mean())
print(df['YearsCode'].median())
print(df['YearsCode'].mode())