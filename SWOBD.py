import networkx as nx
import numpy as np
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

def weight_to_network(weight, threshold):
    weight_bool = weight >= threshold
    G = nx.from_numpy_array(weight_bool)
    return G


def payoff_calculation(G, dict_node, node, payoff_matrix):
    payoff = 0
    nei = list(G.neighbors(node))
    if len(nei) == 0:
        payoff = 0
    else:
        for j in nei:
            payoff += payoff_matrix.loc[dict_node[node]['strategy'][-1], dict_node[j]['strategy'][-1]]
    return payoff


def weight_updating(weight, dict_node, beta, tao, weight_threshold):
    for i in range(len(weight)):
        for j in range(i + 1, len(weight)):
            if dict_node[i]['state'][-1] == dict_node[j]['state'][-1]:
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

row = 10
n_states = row * row
n = 3
weight = np.ones((n * n_states, n * n_states))
G = nx.complete_graph(n * n_states)
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
for i in range(0, n * n_states):
    dict_node[i] = {}
    dict_node[i]['action'] = 'stay'
    dict_node[i]['state'] = [i // n]
    dict_node[i]['q_table'] = build_q_table(n_states, actions)
    dict_node[i]['strategy'] = [np.random.choice(['C', 'D'])]
    dict_node[i]['payoff'] = 0
G_save = [G]
weight_save = [weight]

for time in range(ETime):
    print("\r" + "t:{:.2f}/{:.2f}".format(time, ETime), end = '')
    for agent in dict_node:
        A = choose_action(dict_node[agent]['state'][time], dict_node[agent]['q_table'], delta, n_states, actions)
        S_ = get_action_feedback(dict_node[agent]['state'][time], A, n_states)
        dict_node[agent]['state'].append(S_)
        dict_node[agent]['action'] = A
    weight = weight_updating(weight, dict_node, beta, tao, weight_threshold)
    G = weight_to_network(weight, weight_threshold)
    G_save.append(G)
    weight_save.append(weight)
    for agent in dict_node:
        dict_node[agent]['payoff'] = payoff_calculation(G, dict_node, agent, payoff_matrix)
    for agent in dict_node:
        state = dict_node[agent]['state'][-1]
        state_ = dict_node[agent]['state'][-2]
        A = dict_node[agent]['action']
        agent_nei = list(G.neighbors(agent))
        if len(agent_nei) == 0:
            y = agent
        else:
            y = np.random.choice(agent_nei)
        s_change = dict_node[y]['strategy'][time]
        ran = 1 / (1 + np.exp((dict_node[agent]['payoff'] - dict_node[y]['payoff']) / kappa))
        if 0 <= np.random.uniform(0, 1) < ran:
            dict_node[agent]['strategy'].append(s_change)
        else:
            dict_node[agent]['strategy'].append(dict_node[agent]['strategy'][time])
        reward_direct = get_reward_feedback(state, dict_node)
        q_target = reward_direct + gamma * dict_node[agent]['q_table'].iloc[state, :].max()
        dict_node[agent]['q_table'].loc[state_, A] += eta * (q_target - dict_node[agent]['q_table'].loc[state_, A])