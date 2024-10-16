import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv',  index_col= 'Respondent')#creating index on loading
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)
print(df['ConvertedComp'].median())
print(df.median())
print(df.describe())
print(df['Hobbyist'].value_counts())
print(df['Country'].value_counts(normalize=True))#normalize returns percentage

#Grouping

country_grp = df.groupby(['Country'])
#country_grp.get_group(['United States'])

#gender count by country
print(country_grp['Gender'].value_counts().loc['Trinidad and Tobago'])

#median salary by country
#print(country_grp['ConvertedComp'].median())

#Multiple aggregate
print(country_grp['ConvertedComp'].agg(['median','mean']))

#respondents sing Python
print(country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum()))


#percentage of users who know python by country

respondents = df['Country'].value_counts()

PythonByCountry = country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())

python_df = pd.concat([respondents, PythonByCountry], axis='columns', sort=False)
python_df.rename(columns={'Country':'TotalRespondents','LanguageWorkedWith':'KnowsPython'}, inplace=True)

python_df['PctKnowsPython'] = (python_df['KnowsPython'] / python_df['TotalRespondents'] ) * 100


print(python_df)
