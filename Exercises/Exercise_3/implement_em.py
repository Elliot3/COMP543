import numpy as np
import scipy.stats as st

# One coin has a probability of coming up heads of 0.2, the other 0.6

coinProbs = np.zeros(2)
coinProbs[0] = 0.2
coinProbs[1] = 0.6

# Reach in and pull out a coin numTimes times

numTimes = 100

# Flip it numFlips times when you do

numFlips = 10

# Flips will have the number of heads we observed in 10 flips for each coin

flips = np.zeros(numTimes)

for coin in range(numTimes):
    which = np.random.binomial(1, 0.5, 1)
    flips[coin] = np.random.binomial(numFlips, coinProbs[which], 1)

# Initialize the EM algorithm

coinProbs[0] = 0.79
coinProbs[1] = 0.51

# Define a container to hold the count of heads and tails

res = np.zeros(2)

# Define the EM algorithm

def myEM():

    # Loop through to perform the learning

    for iters in range(20):

        # Calculate the pmf for each coin

        pmf_coin_head = st.binom.pmf(flips, numFlips, coinProbs[0])
        pmf_coin_tail = st.binom.pmf(flips, numFlips, coinProbs[1])

        # Calculate the E-Step results

        e_step_head = pmf_coin_head / (pmf_coin_head + pmf_coin_tail)
        e_step_tail = pmf_coin_tail / (pmf_coin_head + pmf_coin_tail)

        # Get the weighted sums of the heads and tails according to the E-Step

        res[0] = np.sum(e_step_head * flips)
        res[1] = np.sum(e_step_tail * flips)

        # Adjust the probabilities according the the M-Step

        coinProbs[0] = res[0] / np.sum(e_step_head * numFlips)
        coinProbs[1] = res[1] / np.sum(e_step_tail * numFlips)

        # Output the results

        print(coinProbs)

# Run the function and print out the results

print()
print('Results for numFlips = 10')
print()
myEM()

# Change numFlips from 10 to 2

numFlips = 2

# Flips will have the number of heads we observed in 2 flips for each coin

flips = np.zeros(numTimes)

for coin in range(numTimes):
    which = np.random.binomial(1, 0.5, 1)
    flips[coin] = np.random.binomial(numFlips, coinProbs[which], 1)

# Initialize the EM algorithm again

coinProbs[0] = 0.79
coinProbs[1] = 0.51

# Define the EM algorithm again

def myEM():

    # Loop through to perform the learning

    for iters in range(20):

        # Calculate the pmf for each coin

        pmf_coin_head = st.binom.pmf(flips, numFlips, coinProbs[0])
        pmf_coin_tail = st.binom.pmf(flips, numFlips, coinProbs[1])

        # Calculate the E-Step results

        e_step_head = pmf_coin_head / (pmf_coin_head + pmf_coin_tail)
        e_step_tail = pmf_coin_tail / (pmf_coin_head + pmf_coin_tail)

        # Get the weighted sums of the heads and tails according to the E-Step

        res[0] = np.sum(e_step_head * flips)
        res[1] = np.sum(e_step_tail * flips)

        # Adjust the probabilities according the the M-Step

        coinProbs[0] = res[0] / np.sum(e_step_head * numFlips)
        coinProbs[1] = res[1] / np.sum(e_step_tail * numFlips)

        # Output the results

        print(coinProbs)

# Run the function and print out the results

print()
print('Results for numFlips = 2')
print()
myEM()
