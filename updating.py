import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv',  index_col= 'Respondent')#creating index on loading
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)
#column assignment

#df.columns = ['list of column names'] - used to rename all columns

#upper case column names
#df.columns = [ x.upper() for x in df.columns]

#replace strings to underscore in column names

df.columns = df.columns.str.replace(' ','_')

#rename column names
#df.rename(columns={'first_name':'first','lastname':'last'}, inplace=True) -- Key is oldname and value is new name

#updating using loc index
#df.loc[0] = ['Cleveland','Sullivan','salano_cs@yahoo.com']# updates all values for a row

#updating specific columns
#df.loc[0, ['lastname','email']] = ['Badman','badman@yahoo.com']

#update a single value
#df.loc[0, 'last'] = badman
#df.at[0, 'last'] = Sullivan

#return lower case values
#def['emails'].str.lower()

'''
Side note the -- apply function applies a given function to a list of values
eg 1


df['email'].apply(len)

eg 2
def update_email(email):
	return email.upper()

df['email'].apply(update_email)

# using the lamdba function

df['email'].apply(lambda x : x.upper())
eg 2

'''

'''
Applymap - applies to each value in a dataframe, there isn't an applymap for series

eg
df.applymap(len) # applies the len function to each value in the dataframe df
'''

'''
The map method only work on series and subsitute one value for another

df['first'].map({'Cleveland:Simeon','Tanis': 'Simyra'}) # values not substituted are updated with Nan
'''


'''
The replace method only work on series and subsitute one value for another, and leave non subsittuted values as is

df['first'].replace({'Cleveland:Simeon','Tanis': 'Simyra'}) # values not substituted are left unaltered
'''

df.rename(columns={'ConvertedComp':'SalaryUSD'}, inplace=True)
df['Hobbyist'] = df['Hobbyist'].map({'Yes':True, 'No': False})
print(df.head())