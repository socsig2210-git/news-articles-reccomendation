# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import random

# Data Initialisation
k = 5     # number of articles
T = 10000   # horizon (total visits) 
U = 4     # no of user types

# propabilities for each site depending on person category
p_matrix = np.array([[0.8, 0.2, 0.2, 0.2], # p0
                    [0.6, 0.4, 0.4, 0.4], # p1
                    [0.5, 0.5, 0.8, 0.8], # p2
                    [0.4, 0.6, 0.6, 0.6], # p3
                    [0.2, 0.8, 0.8, 0.8]]) # p4

# Shmeiwsi
# px for P(choose article 0) = P(p0 | fem o 25)*P(fem o 25) + P(p0 | male o 25)*P(male o 25) + P(p0 | u 25)*P(u 25)
# P(choose article 0) = 0.8*0.25 + 0.2*0.25 + 0.2*0.5 = 0.35

# visits = np.zeros((T,3)) # visits of type [[botol visi, int person, int news site]]

# for only 1 type of person
# people = np.zeros(T, dtype=np.int64)

# 1 type

# r_cumul = np.zeros(k)
# m_ui = np.zeros(k)
# visits = np.zeros(k, dtype=np.int64)
# ucb = np.zeros(k)
# ucb = np.flip(np.arange(start=2, stop=k+2, dtype=np.float64))

# for i in range(T):
#     person_type = people[i]
    
#     if i < k:
#         site = i
#         ind = i+1
#     else:
#         #TODO how do i calculate which site i use?
#         site = np.argmax(ucb)
#         ind = k

#     p = p_matrix[site][person_type] # chooose i site
#     choice = np.random.choice([1, 0], p=[p, 1-p])
#     visits[site] += 1
#     r_cumul[site] += choice

#     m_ui[site] = r_cumul[site] / visits[site]

#     # update ucb array
#     for j in range(ind):
#         ucb[j] = m_ui[j] + np.sqrt(2*np.log(i+1) / visits[j], dtype=np.float64)

# U types

# TODO: calculate regret during procedure

people = np.random.randint(0,U,T) # 0 -> mo25, 1 -> fo25, 2 -> mu25, 3-> fu25
r_cumul = np.zeros((U,k))
m_ui = np.zeros((U,k))
horizon = np.zeros(U, np.int64)
visits = np.zeros((U, k), dtype=np.int64)
ucb = np.zeros((U,k))

for i in range(T):
    person_type = people[i]
    
    if horizon[person_type] < k:
        site = horizon[person_type]
        ind = site+1
    else:
        #TODO how do i calculate which site i use?
        site = np.argmax(ucb[person_type])
        ind = k

    p = p_matrix[site][person_type] # chooose i site
    choice = np.random.choice([1, 0], p=[p, 1-p])
    horizon[person_type] += 1
    visits[person_type][site] += 1
    r_cumul[person_type][site] += choice

    m_ui[person_type][site] = r_cumul[person_type][site] / visits[person_type][site]

    # update ucb array
    for j in range(ind):
        ucb[person_type][j] = m_ui[person_type][j] + np.sqrt(2*np.log(horizon[person_type]) / visits[person_type][j], dtype=np.float64)


# Print actual rewards
# prop = np.sum(p_matrix, axis=1)[0] 

with open('output.txt', 'w+') as f:
    f.write('Total visits:\n')
    for i, visit in enumerate(horizon):
        f.write(f'Type {i}: {visit}\n')

    f.write('\nUCB\n')
    for i in range(U):
        f.write(f'#For type {i}:\n')
        f.write(f'\nBest site for type {i} is site no. {np.argmax(ucb[i])}\n')
        for j in range(k):
            f.write(f'{j}: ucb: {ucb[i][j]}, m_ui: {m_ui[i][j]}, visits: {visits[i][j]}\n')