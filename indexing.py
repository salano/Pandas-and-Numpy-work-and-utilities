import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv',  index_col= 'Respondent')#creating index on loading
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)
#df.set_index('Country')# the label index can be used with loc to find records
#df.reset_index(inplace=True)
#df.sort_index(ascending=False)# sort index alphabetically/chronologically


#filtering data

high_salary = (df['ConvertedComp'] > 70000)
countries = ['United States','India','United Kingdom','Germany','Canada']

filt = df['Country'].isin(countries)

lamguage_filter = df['LanguageWorkedWith'].str.contains('Python', na=False) #na=False, ignores Nan values

#print(df[high_salary])
#
print(df.loc[lamguage_filter, ['Country','Ethnicity', 'Gender', 'LanguageWorkedWith','ConvertedComp']])