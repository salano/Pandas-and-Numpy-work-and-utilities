import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 100)


df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv')
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)
print(df['Hobbyist'].value_counts())
#print(df.iloc[[0,2], [4, 10]])# get location by integer, first list is rows, second is columns. Index alone can be used
#print(df.loc[0])# uses labels instead of indexes, use case similiar to iloc
print(df.loc[[0,3], ['MainBranch', 'Hobbyist', 'Age']])
print(df.loc[4:9, 'Hobbyist':'Employment'])# Slicing is done with the square brackets and the last index is inclusive
#print(df.shape)
#print(df.head(10))
#print(df.tail(10))
#print(df.info())