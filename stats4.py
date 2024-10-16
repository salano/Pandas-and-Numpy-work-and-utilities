import statistics as stats

# A Probability Distribution finds the probability of different
# outcomes * A coin flip has a probability distribution of .5 *
# A die roll has a probability distribution of 1/6 or .167 *
# When you sum all probabilities you get a value of 1.

# You see here the probabilities of all die rolls with 2 die *
# A Relative Frequency Histogram charts out all those probabilities.
# Pay particular attention to the shape of that chart because...

# Next we'll talk about a Normal Distribution. A Normal Distribution
# is when data forms a bell curve. Also 1 Standard Deviation is
# representative of 68% of the data. 2 standard deviations cover 95%
# and 3 covers 99.7%.

# To have a Normal Distribution the Mean = Median and Mode * Also
# 50% of values are both less than and greater that the mean.

# A Standard Normal Distribution has a mean of zero and a standard
# deviation of 1. If we calculate the mean we see it is 4. If we
# calculate the standard deviation that comes to 1.58.

# We can turn this into a Standard Normal Distribution by subtracting
# the mean from each value and divide by the standard deviation. If
# we do that we get the chart here.

dice_list = [1, 2, 4, 4, 4, 5, 5, 5, 6]
print(f"Sum : {sum(dice_list)}")
print(f"Mean : {stats.mean(*dice_list)}")
print(f"Standard Deviation : {stats.standard_deviation(*dice_list)}")

normalized_list = stats.normalize_list(*dice_list)
print(f"Normal List : {normalized_list}")
print(f"Normal Mean : {stats.mean(*normalized_list)}")
print(f"Normal Standard Deviation : {stats.standard_deviation(*normalized_list)}")

# The Central Limit Theorem states that the more samples you take
# the closer you get to the mean. Also the distribution will
# approximate the Normal Distribution * As you can see as the sample
# size increases the standard deviation decreases.

# The Sample Error measures the accuracy of an estimate. To find
# it divide standard deviation by the square root of the sample
# size. Again notice as the sample size increases the Standard
# Error decreases.

print(f"Standard Error : {stats.sample_error(*normalized_list)}")

# The Z Score gives us the value in standard deviations for the
# percentile we want * For example if we want 95% of the data
# it tells us how many standard deviations are required. * The
# formula asks for the length from the mean to x and divides by
# the standard deviation.

# This will make more sense with an example. Here is a Z Table.
# If we know our mean is 40.8, the standard deviation is 3.5 and
# we want the area to the left of the point 48 we perform our
# calculation to get 2.06. * We then find 2.0 on the left of
# the Z Table * and .06 on the top. * This tells us that the
# area under the curve makes up .98030 of the total.

# Now let's talk about Confidence Intervals. Point Estimates
# are what we have largely used, but they can be inaccurate.
# An alternative is an interval * For example if we had 3
# sample means as you see here we could instead say that
# they lie in the interval of (5,7) * We then state how
# confident we are in the interval. Common amounts are 90%,
# 95% and 99%. For example if we have a 90% confidence that
# means we expect 9 out of 10 intervals to contain the mean *
# Alpha represents the doubt we have which is 1 minus the
# confidence.

# Now I'll show you how to calculate a confidence interval. We
# need a sample mean, alpha, standard deviation and the
# number of samples represented by lowercase n * Here the value
# after the plus or minus represents the Margin of Error.

# Now I'll walk you through an example where we calculate the
# probable salary we would receive if we became a player for
# the Houston Rockets. We have the mean salary * We want our results
# to be confident to 95% * We get alpha from confidence *
# Critical Probability is calculated by subtracting alpha
# divided by 2 from 1. * Then we look up the Z Code in a table.
# If we search for .975 we find that the Z Code is 1.96. *
# We find our standard deviation and then plug in our values. *
# And when we do we find our Confidence Interval salary.

# Calculate the Houston Rockets salary confidence interval
salary_list = [38178000, 37800000, 14057730, 11301219, 8349039, 3540000,
               2564753, 2564753, 2174318, 2028594, 1845301, 903111,
               8111447, 695526, 568422]

# # Formula (x,y) = x̄ ± Z(α/2) * σ/√n
# # x̄ : Sample Mean
# # α : Alpha (1 - Confidence)
# # σ : Standard Deviation
# # n : Sample Size
# get_confidence_interval(sample_mean, alpha, sd, sample_size)
sample_mean = stats.mean(*salary_list)
print(f"Mean {sample_mean}")
confidence = .95
standard_deviation = stats.standard_deviation(*salary_list)
print(f"Standard Deviation {standard_deviation}")

stats.get_confidence_interval(sample_mean, confidence, standard_deviation, 15)