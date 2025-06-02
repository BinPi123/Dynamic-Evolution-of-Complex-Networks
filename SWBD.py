import networkx as nx
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'style':'italic',
        'size': 26,
        }

font2 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 22,
        }
colors=["#0000FF","#000000","#FF0033","#00CC00","#ff7500",'#56004f']
marker_list = ['o','^','s','D','H','>','<']

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns = actions)
    return table

def choose_action(state, q_table, delta, n_states, actions):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > delta) or (state_actions.all() == 0):
        action_name = np.random.choice(actions)
        while True:
            state_new = get_action_feedback(state, action_name, n_states)
            if state_new in np.arange(0, n_states):
                break
            else:
                action_name = np.random.choice(actions)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_action_feedback(state, action, n_states):
    if action == "up":
        state_ = state + row
    if action == "down":
        state_ = state - row
    if action == "left":
        state_ = state - 1
    if action == "right":
        state_ = state + 1
    if action == "stay":
        state_ = state
    if action == "random":
        state_ = np.random.randint(n_states)
    return state_

def get_reward_feedback(state, dict_node):
    reward = 0
    for i in dict_node:
        if dict_node[i]['state'][-1] == state:
            reward += dict_node[i]['payoff']
    return reward


def weight_to_network(weight, threshold, index_array, G):
    edges = G.edges()
    G.remove_edges_from(edges)
    for i in range(len(weight)):
        for j in range(i + 1, len(weight)):
            if weight[i][j] >= threshold:
                G.add_edge(index_array[i], index_array[j])

def payoff_calculation(G, dict_node, node, payoff_matrix):
    payoff = 0
    nei = list(G.neighbors(node))
    if len(nei) == 0:
        payoff = 0
    else:
        for j in nei:
            payoff += payoff_matrix.loc[dict_node[node]['strategy'][-1], dict_node[j]['strategy'][-1]]
    return payoff

def weight_updating(weight, dict_node, beta, tao, weight_threshold, index_array):
    for i in range(len(weight)):
        for j in range(i + 1, len(weight)):
            if dict_node[index_array[i]]['state'][-1] == dict_node[index_array[j]]['state'][-1]:
                if weight[i][j] < weight_threshold:
                    new_weight = tao
                else:
                    new_weight = weight[i][j] / beta + tao
            else:
                if weight[i][j] < weight_threshold:
                    new_weight = 0
                else:
                    new_weight = weight[i][j] / beta
            weight[i][j] = new_weight
            weight[j][i] = new_weight
    for i in range(len(weight)):
        weight[i][i] = 0
    return weight

def information_updating_birth(weight, G, node, weight_threshold, dict_node):
    num_states = [0] * n_states
    for agent in dict_node:
        num_states[dict_node[agent]['state'][-1]] += 1
    birth_state = num_states.index(max(num_states))
    dict_node[node] = {}
    dict_node[node]['action'] = 'stay'
    dict_node[node]['state'] = [birth_state]
    dict_node[node]['q_table'] = build_q_table(n_states, actions)
    dict_node[node]['strategy'] = [np.random.choice(['C', 'D'])]
    dict_node[node]['payoff'] = 0
    connect = []
    for agent in dict_node:
        if dict_node[agent]['state'][-1] == birth_state:
            if agent != node:
                connect.append(agent)
    G.add_node(node)
    for i in connect:
        G.add_edge(node, i)
    index_array = []
    for i in G.degree():
        index_array.append(i[0])
    new_row = np.zeros((1, len(G.nodes) - 1))
    new_col = np.zeros((len(G.nodes), 1))
    weight_new = np.row_stack((weight, new_row))
    weight_new = np.column_stack((weight_new, new_col))
    for i in connect:
        i_index = index_array.index(i)
        weight_new[-1][i_index] = weight_threshold
        weight_new[i_index][-1] = weight_threshold
    return weight_new

def information_updating_death(weight, G, death_node, dict_node, next_death, birth_name):
    index_array_before = []
    for i in G.degree():
        index_array_before.append(i[0])
    G.remove_node(death_node)
    birth_name.append(death_node)
    next_death.pop(death_node)
    dict_node.pop(death_node)
    death_index = index_array_before.index(death_node)
    weight_new = np.delete(weight, death_index, axis = 0)
    weight_new = np.delete(weight_new, death_index, axis = 1)
    return weight_new

def powerlaw(alpha):
    u = random.uniform(0,1)
    result = xmin * math.pow(u, 1 / (1 - alpha))
    return result

lam = 3
xmin = 50
alpha = 3
tnow = 0
tupdate = 1
row = 10
n_states = row * row
n = 2
G = nx.complete_graph(n * n_states)
weight = np.ones((n * n_states, n * n_states))
for i in range(len(weight)):
    weight[i][i] = 0
beta = 1.5
tao = 1
actions = ["up", "down", "left", "right", "stay", "random"]
delta = 0.7
eta = 0.7
gamma = 0.2
weight_threshold = 1
ETime = 5001
r = 0.1
payoff_array = np.array([[1, 1 - r], [1 + r, 0]])
payoff_matrix = pd.DataFrame(payoff_array, columns = ['C', 'D'], index = ['C', 'D'])
kappa = 0.1
dict_node = {}
next_death = {}
for i in range(0, n * n_states):
    dict_node[i] = {}
    dict_node[i]['action'] = 'stay'
    dict_node[i]['state'] = [i // n]
    dict_node[i]['q_table'] = build_q_table(n_states, actions)
    dict_node[i]['strategy'] = [np.random.choice(['C', 'D'])]
    dict_node[i]['payoff'] = 0
    number = powerlaw(alpha)
    next_death[i] = number
next_birth = random.expovariate(lam)
node_num = [n * n_states]
G_num = [G.number_of_nodes()]
G_array = [nx.adjacency_matrix(G).todense()]

birth_name = []
while tnow < ETime:
    print("\r" + "t:{:.2f}/{:.2f}".format(tnow, ETime), end = '')
    while tupdate <= tnow:
        tupdate += 1
        node_num.append(len(dict_node))
        for agent in dict_node:
            A = choose_action(dict_node[agent]['state'][-1], dict_node[agent]['q_table'], delta, n_states, actions)
            S_ = get_action_feedback(dict_node[agent]['state'][-1], A, n_states)
            dict_node[agent]['state'].append(S_)
            dict_node[agent]['action'] = A
        index_array_ = []
        for i in G.degree():
            index_array_.append(i[0])
        weight = weight_updating(weight, dict_node, beta, tao, weight_threshold, index_array_)
        weight_to_network(weight, weight_threshold, index_array_, G)
        G_array.append(np.array(nx.adjacency_matrix(G).todense()))
        G_num.append(G.number_of_nodes())
        for agent in dict_node:
            dict_node[agent]['payoff'] = payoff_calculation(G, dict_node, agent, payoff_matrix)
        Strategy = {}
        for agent in dict_node:
            Strategy[agent] = dict_node[agent]['strategy'][-1]
        for agent in dict_node:
            state = dict_node[agent]['state'][-1]
            state_ = dict_node[agent]['state'][-2]
            A = dict_node[agent]['action']
            agent_nei = list(G.neighbors(agent))
            if len(agent_nei) == 0:
                y = agent
            else:
                y = np.random.choice(agent_nei)
            s_change = Strategy[y]
            ran = 1 / (1 + np.exp((dict_node[agent]['payoff'] - dict_node[y]['payoff']) / kappa))
            if 0 <= np.random.uniform(0, 1) < ran:
                dict_node[agent]['strategy'].append(s_change)
            else:
                dict_node[agent]['strategy'].append(Strategy[agent])
            reward_direct = get_reward_feedback(state, dict_node)
            q_target = reward_direct + gamma * dict_node[agent]['q_table'].iloc[state, :].max()
            dict_node[agent]['q_table'].loc[state_, A] += eta * (q_target - dict_node[agent]['q_table'].loc[state_, A])
    death_node = min(next_death.items(), key = lambda x: x[1])[0]
    death = next_death[death_node]
    birth = next_birth
    if birth < death:
        tnow = birth
        if len(birth_name) == 0:
            node_name = len(next_death)
        else:
            node_name = birth_name[0]
            birth_name.pop(0)
        weight = information_updating_birth(weight, G, node_name, weight_threshold, dict_node)
        number = powerlaw(alpha)
        next_death[node_name] = tnow + number
        next_birth += random.expovariate(lam)
    else:
        tnow = death
        weight = information_updating_death(weight, G, death_node, dict_node, next_death, birth_name)