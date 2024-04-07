# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import random

# Data Initialisation
k = 5     # number of articles
T = 250  # horizon (total visits) 

# propabilities for each site depending on person category
p_matrix = np.array([[0.8, 0.2, 0.2, 0.2], # p0
                    [0.6, 0.4, 0.4, 0.4], # p1
                    [0.5, 0.5, 0.8, 0.8], # p2
                    [0.4, 0.6, 0.6, 0.6], # p3
                    [0.2, 0.8, 0.8, 0.8]]) # p4

# Shmeiwsi
# px for P(choose article 0) = P(p0 | fem o 25)*P(fem o 25) + P(p0 | male o 25)*P(male o 25) + P(p0 | u 25)*P(u 25)
# P(choose article 0) = 0.8*0.25 + 0.2*0.25 + 0.2*0.5 = 0.35

# people = np.random.random_integers(0,3,T) # 0 -> mo25, 1 -> fo25, 2 -> mu25, 3-> fu25
# visits = np.zeros((T,3)) # visits of type [[botol visi, int person, int news site]]

# for only 1 type of person
people = np.zeros(T, dtype=np.int64)

r_cumul = np.zeros(k)
visits = np.zeros(k, dtype=np.int64)
# ucb = np.arange(k)
ucb = np.flip(np.arange(start=2, stop=k+2, dtype=np.float64))

# starting data
for i in range(T):
    person_type = people[i]
    
    # if i < k:
    #     site = i
    #     continue
    # else:
    #     #TODO how do i calculate which site i use?
    #     site = np.argmax(ucb)
    site = np.argmax(ucb)

    p = p_matrix[site][person_type] # chooose i site
    choice = np.random.choice([1, 0], p=[p, 1-p])
    visits[site] += 1
    r_cumul[site] += choice

    m_ui = r_cumul[site] / visits[site]

    # TODO: all ucb should update after each visit
    ucb[site] = m_ui + np.sqrt(2*np.log(i+1) / visits[site], dtype=np.float64)

# Print actual rewards
# prop = np.sum(p_matrix, axis=1)[0] 
for i in range(k):
    print(f'ucb for site no. {i+1}: {ucb[i]}')



# ucb algorithm for specified problem
# p0 = 0.2
# for n in range(T):
#     np.random.choice([1, 0], p=[p0, 1-p0])


# bandit = np.random.random((k,)) # success prob. for each arm
# best = np.amax(bandit) # best arm
# print('best arm = %d' %np.argmax(bandit))