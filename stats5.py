import statistics as stats
import random

# Student's T Distributions are used when your sample
# size is small and/or the population variance is
# unknown

# A T Distribution looks like a Normal Distribution
# with fatter tails meaning a wider dispersion of variables

# When we know the standard deviation we can compute the
# Z Score and use the Normal Distribution to calculate
# probabilities

# The formula is t = (x̅ - μ) / (s/√n), where x̅ is the
# sample mean, μ is the population mean, s is the
# Standard Deviation of the sample and n is the sample
# size

# In this example let's say a manufacturer is promising
# break pads will last for 65,000 km with a .95
# confidence level. * Our sample mean is 62,456.2
# * The standard deviation is 2418.4

# Degrees of freedom is the number of samples taken
# minus 1. If we take 30 samples that means degrees of
# freedom equals 29.

# If we know confidence is .95 then we subtract .95 from
# 1 to get .05. If we look up 29 and .05 in the T Table
# we get a value of 2.045

# If we plug our values into our formula we find the
# interval for our sample.

# Generate Random List between 58000 and 68000

# break_pad_kms = [random.randint(58000, 68000) for i in range(30)]
break_pad_kms = [58500, 58700, 62800, 57220, 62750, 59370, 57720, 60920, 61910, 59260, 63550, 60520, 58710, 57340, 60660, 57750, 60430, 60050, 62970, 58870]
stats.get_t_confidence_interval(.95, *break_pad_kms)

# When used with previous formula you can see results
# are similar
stats.get_confidence_interval(60000, .95, 1988.1782, 20)

# Let's talk about the difference between Dependent &
# Independent Samples. With Dependent samples 1 sample
# can be used to determine the other samples results.
# You'll often see examples of cause & effect or pairs
# of results. An example would be if I roll a die, what
# is the probability that it is odd. Or, if subjects
# lifted dumbbells each day and recorded results before
# and after the week what did we find?

# Independent Samples are those in which samples from
# 1 population has no relation to another group. Normally
# you'll see the word random and not cause and effect
# terms. An example is blood samples are taken from 10
# random people that are tested at lab A. 10 other random
# samples are tested from lab B. Or, Give 1 random group
# a drug and another a placebo and test the results.

# When thinking about probabilities we first must create
# a hypothesis. A hypothesis is an educated guess that
# you can test * If you say restaurants in Los Angeles
# are expensive that is a statement and not a hypothesis
# because there is nothing to test that against * If
# however we say restaurants in Los Angeles are expensive
# versus restaurants in Pittsburgh we can test for that.
# * The technical name for the hypothesis we are testing
# is the Null Hypothesis. An example is a test to see if
# average used car prices fall between $19,000 and
# $21,000 * The Alternative Hypothesis includes all
# other possible prices in this example. That would be
# values from $0 to $19,000 and then from $21,000 and
# higher.

# When you test a hypothesis the probability of
# rejecting the Null Hypothesis when it is actually
# true is called the Significance Level represented by
# α. * Common αs include .01, .05 and .1. * Previously
# we talked about Z Tables. If the sample mean and
# the population mean are equal then Z equals 0. * If
# we create a bell graph and we know that α is .05
# then we know that the rejection for the Null
# Hypothesis is found at α/2 or .025. * If we use a
# Z Table and we know µ is 0 and α/2 = .025 we find that
# the rejected region is less than -1.96 and greater
# than 1.96. (This is known as a 2 sided test)

# * With 1 sided tests for example if I say I think
# used car prices are greater than $21,000, the
# Null Hypothesis is everything to the right of the
# Z Code for α instead of α/2 which is 1 - .05 = .95
# In the Z Table that is -1.65.

# When it comes to hypothesis errors there are 2 types
# Type I Errors called False Positives, refer to a
# rejection of a true null hypothesis. The probability
# of making this error is alpha. * Then you have Type
# II errors called false negatives which is when you
# accept a false null hypothesis. This error is
# normally caused by poor sampling. The probability
# of making this error is represented by Beta
# * The goal of hypothesis testing is to reject
# a false null hypothesis which has a probability
# of 1 - Beta. You increase the power of the test by
# increasing the number of samples.

# This example will clear hypothesis errors up. If you
# believe the null hypothesis is that there is no
# reason to apply for a job because you won't get it.
# You can call this the status quo belief. * If you then
# don't apply and the null hypothesis was correct
# you'd see that your decision was correct. * Also if
# you rejected the null hypothesis and applied and you
# got the job you would see again that you made the
# correct decision. * However if the hypothesis was
# correct and you applied that would be an example of a
# Type I Error. * And again if you choose not to apply
# but the hypothesis was false this would be an example
# of a Type II Error