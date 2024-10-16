import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_public.csv',  index_col= 'Respondent')#creating index on loading
schema_df = pd.read_csv('D:\\Python\\Pandas\\survey\\survey_results_schema.csv')

print(df.columns)

#add column
#df['fullname'] = #df['First']+ ' '+#df['Last']

#drop column
#df.drop(columns=['first','last'], inplace=True)

#split a column

#df['fullname'].str.split(' ', expand=True)

#df[['first','last']] = df['fullname'].str.split(' ', expand=True)


#Add columns

#df.append({'first':'Jayedn','last':'Sullivan'}, ignore_index=True)


#append two dataframes
#df = df.append(df2, ignore_index=True)

#drop rows

#df.drop(index=4) removes 5th row

#filt=['last']=='Joe'
#df.drop(index=df[filt].index) 



#Sorting data
#df.sort_values(by='last', ascending=False)
#df.sort_values(by=['last','first'], ascending=False)
#df.sort_values(by=['last','first'], ascending=[False, True], inplace=True)
#df.sort_index()#resets sorting

df.sort_values(by=['Country','ConvertedComp'], ascending=False, inplace=True)
#print(df.head(20))

#print(df['ConvertedComp'].nlargest(10))
#print(df.nlargest(10,'ConvertedComp'))
print(df['ConvertedComp'].nsmallest(10))

