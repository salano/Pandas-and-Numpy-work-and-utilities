import statistics as stats

# I want to calculate if my sample is higher or
# lower than the population mean. To find out I
# need a 2 sided test. The population mean is the
# Null Hypothesis. * That Null Hypothesis is that
# break pads should last for 64,000 kms.

population_mean = 64000

# * Here is my Sample break pad data
break_pad_kms = [58500, 58700, 62800, 57220, 62750, 59370, 57720, 60920, 61910, 59260, 63550, 60520, 58710, 57340, 60660, 57750, 60430, 60050, 62970, 58870]

# We calculate our sample mean * standard deviation
# * sample size, * Sample Error

# * We need to standardize our means so we can compare
# them even if they have different standard deviations.
# * We standardize our variable by subtracting
# the mean and then divide by the standard deviation.
# When we do this we normalize our data meaning we
# get a mean of zero and a standard deviation of 1.
# Z = (x̅ - μ0) / Sample Error
# Sample Error = standard_deviation(*args) /
# (math.sqrt(len(args)))

# * We then get the absolute value of this result

sample_mean = stats.mean(*break_pad_kms)
sample_sd = ("%.2f" % stats.standard_deviation(*break_pad_kms))
sample_error = ("%.2f" % stats.sample_error(*break_pad_kms))

print(f"Mean : {sample_mean}")
print(f"Standard Deviation : {sample_sd}")
print(f"Sample Size : {len(break_pad_kms)}")
print(f"Sample Error : {sample_error}")

z_score = (sample_mean - population_mean) / float(sample_error)
print(f"Z Score : {z_score}")

# If my confidence is .95 α is .05 and since we are
# using a 2 sided test we use α/2 = .025. * If we
# subtract .025 from 1 we get .9750. * If we look up
# .9750 on the Z Table we get a Z Score of 1.96.

# * We now compare the absolute value of the z score we
# calculated before which is 8.99 to the Critical
# Value which is 1.96. If 8.99 is greater than 1.96
# which it is we reject the Null Hypothesis. To be
# more specific we are saying at .95 confidence level
# we reject that the break pads have an average
# lifecycle of 64,000 km.

# The P Value is the smallest level of significance
# at which we can reject the Null Hypothesis. * In
# our example we found a Z Score of 8.99 which isn't
# on our chart. * Let's say instead that the Null
# Hypothesis was 61,750 kms. That would mean the
# hypothesis would be correct at 1 - .99996 = .00004
# significance. So here the P Value for a 1 sided
# test is .00004. For a 2 sided test we multiply
# .00004 by 2 = .00008.

population_mean = 61750
z_score = (sample_mean - population_mean) / float(sample_error)
print(f"Z Score : {z_score}")