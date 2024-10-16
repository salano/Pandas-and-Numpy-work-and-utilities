import statistics as stats
import random
from functools import reduce

# Regression Analysis is used to examine the
# relationship between 2 or more variables.
# * You use it to determine which factors matter
# the most * and which factors can be ignored.
# * The Dependent Variable is what you want to
# better understand. Independent Variables are
# what does or doesn't effect the Dependent
# Variable.

# * For example if we had a movie theater
# and we wanted to improve customer satisfaction
# customer satisfaction would be the dependent
# variable. The independent variables that effect
# it may be sound quality, picture quality, the
# seat comfort, the quality of the food or price.

# Simple Linear Regression Model
# y = β0 + β1 * x1 + ε
# y : Dependent Variable (What we are trying to predict) ŷ is used for predicted values
# x1, ... xn : Independent Variables
# β0 : A constant that represents the minimum
# value of y (Y Intercept)
# β1 : The coefficient that quantifies the effect
# of the independent variable x1 on y
# ε : Represents estimation errors which would
# be the difference between the observed
# value of y and the value for y that the regression
# predicts. The average value should be 0.

# We do this like we do with any linear equation. We find the slope and then
# b0 which is the Y intercept. We are basically averaging the sample points
# to our line. This is called the regression line. We note that it is a
# regression line by using y hat instead of y.

# Here is the formula for calculating b1. We sum all values of x minus
# their means and the same for all values of y. We square the results
# to eliminate negative values. We then divide by the sum of x minus
# the mean again squared. Now we have the slope. To calculate the
# y intercept or b0 I find y bar - slope * x bar.

# Here is an example on how you'd calculate the linear regression
# line. Get the means for x & y. Sum the product of each value of
# x minus the mean and the same for y. Get the sum of all values of
# x minus the mean squared. Then find the slope by dividing those
# values to get 5.958. The calculate the value for the y intercept.
# Then you can create the formula for the line which you can see
# to the right.

# How do we find out if our regression line is a good fit for our
# data? We do that with something we have already covered which is
# the correlation coefficient. Remember that the correlation
# coefficient calculates whether the values of x and y are
# related (correlated). We calculate it by finding the covariance
# of X & Y and then divide by the product of the standard deviations
# of X & Y. If the value is close to 1 then the data is highly
# correlated which means our regression line should have an easy
# to modeling the data.

# Let's work through an example where we find the correlation
# coefficient. First we must calculate the covariance for all
# x and y values which equals 1733.09.

# Now that we have the covariance we can divide it by the
# standard deviation of x multiplied by the standard deviation
# of y. When we do that we get .9618. Since .9618 is so close to
# 1 we know that are linear regression line will be tightly
# matched to the data.

temp_sales_day_list = [[37, 292], [40, 228], [49, 324], [61, 376], [72, 440], [79, 496], [83, 536], [81, 556], [75, 496], [64, 412], [53, 324], [40, 320]]

temp_sales_sep_list = [[37, 40, 49, 61, 72, 79, 83, 81, 75, 64, 53, 40], [292, 228, 324, 376, 440, 496, 536, 556, 496, 412, 324, 320]]

print("Linear Regression List")
print(stats.get_linear_regression_list(temp_sales_day_list))
print(f"Correlation Coefficient : {stats.correlation_coefficient(temp_sales_sep_list)}")
print()


# Generates a random list that adds up to the sum provided n
# using the defined number of values num_terms
def random_list_defined_sum(goal_sum, num_values):
    # Generate a random sample with values in the range
    # between 1 to the target sum and add 0 and the goal
    # sum to the endpoints of the list
    a = random.sample(range(1, goal_sum), num_values) + [0, goal_sum]
    # Sort the list
    list.sort(a)
    # If you subtract successive values in the list the resulting
    # list will have the defined goal sum
    return [a[m+1] - a[m] for m in range(len(a) - 1)]


# Generates a random list that will have an average value
# of expected_avg, a defined list length list_length, will have a
# minimum value a and a maximum value of b
def random_list_defined_avg(expected_avg, list_length, a, b):
    while True:
        # Generate random list with values between min and max
        rand_list = [random.randint(a, b) for i in range(list_length)]
        # Find averages for the list until we get our target average
        # and then return the list
        avg = reduce(lambda x, y: x + y, rand_list) / len(rand_list)
        if avg == expected_avg:
            return rand_list


# Define list of average temperatures I'll use for testing
# linear regression
temp_list = [37, 40, 49, 61, 72, 79, 83, 81, 75, 64, 53, 40]
# Used to generate fake sales with defined sums for testing
# linear regression
sales_list = [292, 228, 324, 376, 440, 496, 536, 556, 496, 412, 324, 320]
# List that will hold all temperature values generated which
# will be used to calculate the Correlation Coefficient
gen_temp_list = []
# List that will hold all the generated sales for each day
# that I'll use to calculate the Correlation Coefficient
gen_sales_list = []
# Will hold both gen_temp_list and gen_sales_list
gen_sales_temp_list = []


# Will generate all of the fake temperature and sales data
# that I can use to demonstrate both linear regression and
# how Correlation Coefficient can define if a linear
# regression is a good fit for our sample points
def get_temp_sales_list():
    new_list = []
    # Generate 12 months of temp and sales lists
    for i in range(12):
        new_temp_list = random_list_defined_avg(temp_list[i], 4, temp_list[i]-2, temp_list[i]+2)
        new_sales_list = random_list_defined_sum(sales_list[i], 4)
        # Generate 28 days worth of temp and sales lists for
        # each month
        for j in range(4):
            # Make individual lists for just daily temp and
            # sales and append those lists
            day_data_list = [new_temp_list[j], new_sales_list[j]]
            new_list.append(day_data_list)

            # Add to list of just temps and sales
            gen_temp_list.append(new_temp_list[j])
            gen_sales_list.append(new_sales_list[j])
    return new_list


ice_cream_list = get_temp_sales_list()

lr_list = stats.get_linear_regression_list(ice_cream_list)
# print(lr_list)

# print(gen_temp_list)
# print(gen_sales_list)

# Contains x and y values for lr_list
x_y_ic_list = [gen_temp_list, gen_sales_list]

# print(f"Ice Cream Correlation Coefficient : {stats.correlation_coefficient(x_y_ic_list)}")