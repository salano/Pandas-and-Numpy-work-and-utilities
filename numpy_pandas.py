# Provides ways to work with large multidimensional arrays
import numpy as np 
# Allows for further data manipulation and analysis
import pandas as pd

# In Anaconda -> Environments -> Not Installed -> pandas-datareader -> Apply
#from pandas_datareader import data as web # Reads stock data 
import matplotlib.pyplot as plt # Plotting
import matplotlib.dates as mdates # Styling dates

#plotly imports
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go



import warnings
warnings.simplefilter("ignore")

#dataframe display parameters
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Define path to files
# For MacOS
# PATH = "/Users/derekbanas/Documents/Tutorials/Python for Finance/"
# For Windows
PATH = r"C:\python_files"


# Make an Numpy array
l1 = [1,2,3,4,5]
npa1 = np.array(l1)
print(npa1)



'''
Numpy Crash Course
NumPy is an amazing scientific computing library that is used by numerous other Python Data Science libraries. It contains many mathematical, array and string functions that are extremely useful. Along with all the basic math functions you'll also find them for Linear Algebra, Statistics, Simulation, etc.

NumPy utilizes vector (1D Arrays) and matrice arrays (2D Arrays).
'''


# Creates array from 0 to 4
npa2 = np.arange(0,5)
print(npa2)


# Define a step
npa3 = np.arange(0,10,2)
print(npa3)

# Create a 4 row matrix with 3 columns with all having a value of 1
mat1 = np.ones((4,3))
print(mat1)

# Create a 4 row matrix with 3 columns with all having a value of 0
mat2 = np.zeros((4,3))
print(mat2)

mat3 = np.random.randint(0,50,(4,3))
print(mat3)


# Generate 10 equally distanced values between 1 and 10
mat4 = np.linspace(1,10,10)
print(mat4)

# Create array with 12 values
mat6 = np.random.randint(0,50,12)
print(mat6)

# Reshape to a 3 row 4 column array
mat6 = mat6.reshape(3,4)
print(mat6)

# Reshape into a 3D array with 3 blocks, 2 rows, 2 columns
mat7 = mat6.reshape(3,2,2)
print(mat7)

# Reshape into a 3D array with 2 blocks, 3 rows, 2 columns
mat8 = mat6.reshape(2,3,2)
print(mat8)


# Get the value in the 2nd block, 3rd row and 1st column
print(mat8[1,2,0])

# Provide a boolean array where values are above 20
print(mat6)
print(mat6 > 20)


# Generate 50 random values between 0 and 100
mat5 = np.random.randint(0,100,50)
print(mat5)

print("Mean :", mat5.mean())
print("Standard Deviation :", mat5.std())
print("Variance :", mat5.var())
print("Min :", mat5.min())
print("Max :", mat5.max())


# Used when you want to replicate randomization
np.random.seed(500)
mat9 = np.random.randint(0,50,10)
print(mat9)

# Everything goes back to random on the next call
mat10 = np.random.randint(0,50,10)
print(mat10)







'''
Pandas
Pandas provides numerous tools to work with tabular data like you'd find in spreadsheets or databases. It is widely used for data preparation, cleaning, and analysis. It can work with a wide variety of data and provides many visualization options. It is built on top of NumPy.
'''

#Read Data from a CSV

def get_df_from_csv(ticker):
    try:
        df = pd.read_csv(PATH + ticker + '.csv', index_col='Date', 
                         parse_dates=True)
    except FileNotFoundError:
        pass
        # print("File Doesn't Exist")
    else:
        return df


msft_df = get_df_from_csv("MSFT")
print(msft_df)



#Read Data from Excel

def get_df_from_excel(file):
    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        pass
        print("File Doesn't Exist")
    else:
        return df


 # You may have to run this in the Qt Console : pip install openpyxl
file = PATH + "Wilshire-5000-Stocks.xlsx"
w_stocks = get_df_from_excel(file)
print(w_stocks)

'''
#Read Data from HTML
g_data = pd.read_html("https://en.wikipedia.org/wiki/List_of_current_United_States_governors")
print(g_data)



# We can define that we want the 2nd table on the page
g_data = pd.read_html("https://en.wikipedia.org/wiki/List_of_current_United_States_governors")[1]
print(g_data)

'''

# You can also search for phrases in the table
d_data = pd.read_html("https://en.wikipedia.org/wiki/Demographics_of_the_United_States", 
                      match="Average population")[0]
print(d_data)


#Replace Spaces in Column Names
d_data.columns = [x.replace(' ', '_') for x in d_data.columns]
print(d_data)

#Remove Characters in Columns

# Remove parentheses and whats inside them
d_data.columns = d_data.columns.str.replace(r"\(.*\)","",  regex=True)
# Remove brackets and whats inside them
d_data.columns = d_data.columns.str.replace(r"\[.*\]","" , regex=True)
print(d_data)


#Rename Columns
# You could add additional with commas between {}
d_data = d_data.rename(columns={'Unnamed:_0': 'Year'})
print(d_data)


#Remove Characters in Columns
# Removes brackets and what is inside for whole column
d_data.Year = d_data.Year.str.replace(r"\[.*\]","" , regex=True)
print(d_data)

#Select Columns
print(d_data.Live_births)
print(d_data['Deaths'])


#Make a Column an Index
d_data.set_index('Year', inplace=True)
print(d_data)

#Grab Data from Multiple Columns
print(d_data[["Live_births", "Deaths"]])

#Grab a Row
print(d_data.loc['2020']) #by index
print(d_data.iloc[85]) #recod count #/row number

#Add a Column
# Create a column showing population growth for each year
d_data['Pop_Growth'] = d_data["Live_births"] - d_data["Deaths"]
print(d_data)

#Delete Column
d_data.drop('Pop_Growth', axis=1, inplace=True)
print(d_data)


#Delete a Row
d_data.drop('1935', axis=0, inplace=True)
print(d_data)


#Manipulating Data
c_data = pd.read_html("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)")[2]
print(c_data)

#print column names
for col in c_data.columns:
    print(col)


# Delete a level of a multilevel column name
c_data.columns = c_data.columns.droplevel()
print(c_data)


# Keep only columns if they haven't used the same name prior
c_data = c_data.loc[:,~c_data.columns.duplicated()]
print(c_data)


# Delete any rows with NaN values by taking only rows that don't contain NaNs
c_data = c_data[c_data['Estimate'].notna()]# all row with nan data
print(c_data)


# Remove []s and what is in them in Year column
# Removes brackets and what is inside for whole column
c_data.Year = c_data.Year.str.replace(r"\[.*\]","" , regex=True)
print(c_data)

# Rename country column
c_data.rename(columns={"Country/Territory": "Country", "Estimate": "GDP", "UN region":"Region"}, 
              inplace=True)
print(c_data)



# Remove * in Country column
c_data.Country = c_data.Country.str.replace("*","")
print(c_data)


'''
# Groupby allows you to group rows based on a column and perform a function
# Mean GDP by region
print(c_data.groupby("Region").mean())


# Median GDP by region
print(c_data.groupby("Region").median())
'''

# Dictionary with ice cream sales data
dict1 = {'Store': [1,2,1,2], 'Flavor': ['Choc', 'Van', 'Straw', 'Choc'], 
         'Sales': [26, 12, 18, 22]}

# Convert to Dataframe
ic_data = pd.DataFrame(dict1)
'''
# Group data by the store number
by_store = ic_data.groupby('Store')
# Get mean sales by store
print(by_store.mean())

# Get sales total just for store 1
print(by_store.sum().loc[1])

# You can use multiple functions of get a bunch
print(by_store.describe())
'''

'''
Plotly
Plotly allows you to create over 40 beautiful interactive web-based visualizations that can be displayed in Jupyter notebooks or saved to HTML files. It is widely used to plot scientific, statistical and financial data.

You can install using Anaconda under the environment tab by searching for Plotly. You'll also need Cufflinks and a few other packages that you can install by running : conda install -c conda-forge cufflinks-py in your command line or terminal. Also you can use the commands pip install plotly and pip install cufflinks. Cufflinks connects Plotly to Pandas.

https://github.com/derekbanas/Python4Finance/blob/main/Numpy_Pandas.ipynb



'''

# Plot the value of a dollar invested over time
# Use included Google price data to make one plot
df_stocks = px.data.stocks()
px.line(df_stocks, x='date', y='GOOG', labels={'x':'Date', 
                                               'y':'Value of Dollar'})





# Make multiple line plots
fig = px.line(df_stocks, x='date', y=['GOOG','AAPL'], labels={'x':'Date', 
                                                        'y':'Value of Dollar'},
       title='Apple Vs. Google')

fig.show()
