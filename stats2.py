import csv
import statistics as stats
from pprint import pprint



Data_path = r"D:\Python\data\computersales.csv"

# Pulls data on sales from a csv file

with open(Data_path, newline="") as csv_file:
    reader = csv.reader(csv_file)
    sales_data = list(reader)

#pprint(sales_data) 

def get_string_data_from_csv(num_rows,index):
    # Will hold column data pulled from CSV file
    data_list = [0]*num_rows  #num_rows represents the number of rows in data set

    # Dictionary that will hold string and count
    # of that string in the list
    data_dict = {}

    # Get items in question from CSV file
    for i in range(1, len(sales_data)): #0 index is column names which we need to ignore
        data_list[i] = sales_data[i][index]
   

    # Delete 1st index with no data
    del data_list[0]
    # Convert to a set to get only unique values
    data_set = set(data_list)
    unique_list = list(data_set)

    for i in range(0, len(unique_list)):
        # Get number of times item shows in the list
        # using string stored in unique_list
        num_of_items = data_list.count(unique_list[i])

        # Add key and value to the dictionary
        data_dict[unique_list[i]] = num_of_items

    return data_dict

#print(get_string_data_from_csv(40, 3))

def get_key_profit_list(index, data_dict, profit_index):
    profit_dict = {}
    for key in data_dict.keys():
        # Create dictionary key with a list
        profit_dict[key] = []
        for i in range(1, len(sales_data)):
            if key == sales_data[i][index]:
                # add profit to list in dictionary
                profit_dict.setdefault(key, []).append(float(sales_data[i][profit_index]))
    return profit_dict

category_dict = get_string_data_from_csv(40, 3)


#print(f" category list : {get_key_profit_list(3,category_dict,13)}")
#exit()

def get_profit_mean_category_dict(data_dict):
    # Holds dict key and mean profit for each key
    profit_mean_dict = {}
    # Creates a dict with key and mean value for each category
    for key in data_dict.keys():
        profit_mean_dict[key] = stats.mean(*data_dict[key])
    return profit_mean_dict

key_profit_list = get_key_profit_list(3,category_dict,13)
#print(f" key Profit Mean : {get_profit_mean_category_dict(key_profit_list)}")


# NEW Get variance for all categories
def get_standard_deviation(data_dict):
    # Holds dict key and mean profit for each key
    sd_dict = {}
    # Creates a dict with key and mean value for each category
    for key in data_dict.keys():
        sd_dict[key] = stats.standard_deviation(*data_dict[key])
    return sd_dict


# NEW Gets coefficient variation for all categories
def get_coefficient_variation(data_dict):
    # Holds dict key and mean profit for each key
    cv_dict = {}
    # Creates a dict with key and mean value for each category
    for key in data_dict.keys():
        cv_dict[key] = stats.coefficient_variation(*data_dict[key])
    return cv_dict


def get_mean_profit_data(title, index):
    print(title + " Data")
    category_dict = get_string_data_from_csv(index)
    # Print number in each category
    print(category_dict)
    # Print categories with all profits as a list
    print(f"{title} Profit List : {get_key_profit_list(index, category_dict)}")
    # Print categories with the mean of profits
    print(f"{title} Profit Mean : {get_profit_mean_category_dict(get_key_profit_list(index, category_dict))}")

    # NEW Get Variance
    print(f"{title} Standard Deviation : {get_standard_deviation(get_key_profit_list(index, category_dict))}")

    # NEW Get Coefficient Variation
    print(f"{title} Coefficient Variation : {get_coefficient_variation(get_key_profit_list(index, category_dict))}\n")


# Gets sum of categories, sales per category and
# mean profits for categories
'''
revise to make more generic
get_mean_profit_data("Sex", 3)
get_mean_profit_data("State", 6)
get_mean_profit_data("Product Company", 7)
get_mean_profit_data("Product Type", 9)
get_mean_profit_data("Lead Source", 14)
get_mean_profit_data("Sale Month", 15)
get_mean_profit_data("Sale Year", 16)
'''


# Gets the index I'm searching in the CSV file along with
# an array with the array steps from 1st max on up.
# Example [10, 20, 30] would grab ranges 0-10, 20-29, 30-100
def get_range_data(index, max_rng_list):
    # Creates the dictionary with dynamically created keys that define
    # the ranges I want to grab
    # If supplied with [29, 39] it makes {'rng_0_29': 0, 'rng_30_39':0}
    range_dict = {}
    range_index = 0
    for i in range(0, len(max_rng_list)):
        rng_key = "rng_" + str(range_index) + "_" + str(max_rng_list[i])
        range_dict[rng_key] = 0
        range_index = int(max_rng_list[i]+1)

    # Cycle through the dictionary keys and get the high and low
    # range
    for key in range_dict.keys():
        rng_list = key.split("_")
        low_range = rng_list[1]
        high_range = rng_list[2]

        # Cycle through data in the supplied index while searching for
        # matches in range defined in the dictionary keys
        for i in range(1, len(sales_data)):
            if int(low_range) < int(sales_data[i][index]) <= int(high_range):
                range_dict[key] += 1

    return range_dict


# Define the maximums I want for each range to search for
# in sales_data
my_list = [29, 39, 49, 80]
age_dict = get_range_data(4, my_list)
print(age_dict)


def get_range_profit_list(index, data_dict, profit_index):
    profit_dict = {}
    for key in data_dict.keys():
        # Create dictionary key with a list
        profit_dict[key] = []

        # Get range data used to search in sales_data
        rng_list = key.split("_")
        low_range = rng_list[1]
        high_range = rng_list[2]

        for i in range(1, len(sales_data)):
            if int(low_range) < int(sales_data[i][index]) <= int(high_range):
                # add profit to list in dictionary
                profit_dict.setdefault(key, []).append(float(sales_data[i][profit_index]))
    return profit_dict

profit_range_list = get_range_profit_list(4, age_dict, 13)


def get_range_profit_mean_category_dict(index, data_dict, profit_index):
    # Holds dict key and mean profit for each key
    profit_mean_dict = {}
    # Creates a dict with key and mean value for each category
    for key in data_dict.keys():

        # Create dictionary key with a list
        profit_mean_dict[key] = []

        # Get range data used to search in sales_data
        rng_list = key.split("_")
        low_range = rng_list[1]
        high_range = rng_list[2]

        for i in range(1, len(sales_data)):
            if int(low_range) < int(sales_data[i][index]) <= int(high_range):
                # add profit to list in dictionary
                profit_mean_dict.setdefault(key, []).append(float(sales_data[i][profit_index]))

        # Get mean of list and store the mean of the
        # list instead of the list of values
        profit_mean_dict[key] = stats.mean(*profit_mean_dict[key])

    return profit_mean_dict

range_profit_list = get_range_profit_mean_category_dict(4, age_dict, 13)





# NEW Gets all data in the list for the chosen index
def get_category_list(index):
    data_list = [0] * 40
    # Get all values for provided index
    for i in range(1, len(sales_data)):
        data_list[i] = sales_data[i][index]
    # Delete 1st index with no data
    del data_list[0]
    return data_list


# Gets profit list in order and not sorted
profit_list = [0] * 40
del profit_list[0]
profit_list = get_category_list(13)
print(profit_list)