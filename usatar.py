import numpy as np
from numpy import random
import pandas as pd
from queue import PriorityQueue
        

def graph_creation():
    node_list = {}

    # state defines if that node is empty(-1), has the predator(-10), has the agent(1) or has the prey(10)
    # initially all the noeds are empty
    state = -1

    # making initial connections for each nodes
    for x in range(50):
        connections = []
        if x == 0:
            connect2 = 49
            connections.append(connect2)
        else:
            connect2 = x - 1
            connections.append(connect2)
        
        if x == 49:
            connect1 = 0
            connections.append(connect1)
        else:
            connect1 = x + 1
            connections.append(connect1)

        node_list[x] = {'state' : state, 'connections' : connections, 'degree' : len(connections) }

    y = list(np.arange(50))
    # Making additional connections to each node until the degree is 3 or no other connections are possible
    while y:
        x = random.choice(y)
        x_neighbour = findNeighbor(x)
        z = random.choice(x_neighbour)
        if node_list[z]['degree'] < 3 and node_list[x]['degree'] < 3:
            node_list[z]['degree'] += 1
            node_list[x]['degree'] += 1
            update1 = node_list[x]['connections']
            update1.append(z)
            node_list[x]['connections'] = update1
            update2 = node_list[z]['connections']
            update2.append(x)
            node_list[z]['connections'] = update2
        y.remove(x)

    # making additional list of connections of each nodes
    connection_list = []
    for x in  range(50):
        connection_list.append(node_list[x]['connections'])
    #print(connection_list)

    return node_list, connection_list
    

# finds the neighbours of that node
def findNeighbor(x):
    for_n = []
    for i in range(5):
        element = x + i + 1
        if element == 50:
            element = 0
            for_n.append(element)
        elif element > 50:
            element = element % 49 - 1
            for_n.append(element)
        else:
            for_n.append(element)
    back_n = []
    for j in range(5):
        element = x - (j + 1)
        if element < 0:
            element = 50 - abs(element)
            back_n.append(element)
        else:
            back_n.append(element)

    N_list = back_n[::-1] + for_n
    N_list.remove(N_list[4])
    N_list.remove(N_list[4])
    return N_list


# Finding the distance between 2 nodes using dijkstra algorithm
def dijkstra(start, end, list1):
    count = 0
    last_pos = {}
    g = list(np.zeros(50))
    f = list(np.zeros(50))
    for x in range(50):
        g[x] = float("inf")
    g[start] = 0
    for x in range(50):
        f[x] = float("inf") 
    f[start] = 0
    kyu = PriorityQueue()
    kyu.put((f[start], count, start))
    h_map = {start}
    current = []
    while (len(h_map) > 0):
        current = kyu.get()[2]
        h_map.remove(current)
        node_connections = list1[current]
        for x in node_connections:
            temp_g = g[current] + 1
 
            if (temp_g < g[x]):
                last_pos[x] = current
                g[x] = temp_g
                f[x] = temp_g
                if x not in h_map:
                    count += 1
                    kyu.put((f[x], count, x))
                    h_map.add(x)
    path = []
    last_c = end
    while(last_c in last_pos):
        path.append(last_c)
        last_c = last_pos[last_c]
    
    path = path[::-1]
    return path

# Method that decides predator movement
def predator_movement(predator_position, agent_position, list1):
    predator_neighbour = list1[predator_position]
    path_size = []
    for x in predator_neighbour:
        pred_path = dijkstra(x, agent_position, list1)
        path_size.append(len(pred_path))
    priority_max = min(path_size)
    move_to_list = []
    for y in range(len(path_size)):
        if path_size[y] == priority_max:
            move_to_list.append(predator_neighbour[y])
    move_to = random.choice(move_to_list)
    rand_pos_moves = list1[predator_position]
    rand_pos_move = random.choice(rand_pos_moves)
    new_pos = random.choice([rand_pos_move, move_to], p=[0.4, 0.6])
    return new_pos

def pred_tarns_prob(predator_position, agent_position, list1,pos):
    predator_neighbour = list1[predator_position]
    path_size = []
    for x in predator_neighbour:
        pred_path = dijkstra(x, agent_position, list1)
        path_size.append(len(pred_path))
    priority_max = min(path_size)
    move_to_list = []
    for y in range(len(path_size)):
        if path_size[y] == priority_max:
            move_to_list.append(predator_neighbour[y])
    # move_to = random.choice(move_to_list)
    rand_pos_moves = list1[predator_position]
    if pos in move_to_list:
        prob = (1/len(move_to_list)) * 0.6
    else:
        prob = (1/len(move_to_list)) * 0.6 + (1/len(rand_pos_moves)) * 0.4
    return prob

# Method that decides predator movement
def prey_movement(prey_position, list1):
    nlist = list1[prey_position]
    new_pos = random.choice(nlist)
    return new_pos

def prey_trans_prob(prey_position, list1):
    nlist = list1[prey_position]
    prob = 1/(len(nlist) + 1)
    return prob

# initializing utility for each state
def initU(state, connections):
    agent_pos = state[0]
    prey_pos = state[1]
    path = dijkstra(agent_pos, prey_pos, connections)
    dist = len(path)
    return dist


# for x in range(50):
#     for y in range(50):
#         for z in range(50):
#             state = (x,y,z)
#             if (x == y):
#                 ustar[state] = 0
#                 continue
#             if (x == z):
#                 ustar[state] = float('inf')
#                 continue
#             ustar[(x,y,z)] = initU(state, connections)

# # setting reward as one
# reward = 1
# new_ustar = ustar.copy()
# iterations = 0
# count = 0
# while count != 125000:
#     count = 0
#     for state in ustar:
#         if (state[0] == state[1]):
#             new_ustar[state] = 0
#             loss = abs(new_ustar[state] - ustar[state])
#             if loss <= 0.001:
#                 count += 1
#             continue
#         if (state[0] == state[2]):
#             new_ustar[state] = float('inf')
#             loss = abs(new_ustar[state] - ustar[state])
#             if loss <= 0.001:
#                 count += 1
#             continue
#         util = []
#         a_actions = connections[state[0]]
#         pr_actions = connections[state[1]]
#         pd_actions = connections[state[2]]
#         for x in a_actions:
#             summation = 1
#             for i in pr_actions+[state[1]]:
#                 pr_prob = prey_trans_prob(state[1], connections)
#                 for j in pd_actions:
#                     curr_state = (x, i, j)
#                     pd_prob = pred_tarns_prob(state[2], x, connections, j)
#                     summation += ustar[curr_state] * pr_prob * pd_prob
#             utility = summation
#             util.append(utility)
#         new_ustar[state] = min(util)
#         loss = abs(new_ustar[state] - ustar[state])
#         if loss <= 0.001:
#             count += 1
#     ustar = new_ustar.copy()
#     iterations += 1
#     print(iterations)

# fields =  ["Agent", "Prey", "Predator", "UStar"]   
# data = pd.DataFrame(columns= fields)
# for x in ustar:
#     values = pd.DataFrame([{"Agent" : x[0], "Prey" : x[1], "Predator" : x[2], "UStar" : ustar[x]}])
#     data = pd.concat([data, values], ignore_index=True)
# data.to_csv('UStar.csv', index=False)




connections = [[49, 1, 2], [0, 2, 4], [1, 3, 0], [2, 4], [3, 5, 1], [4, 6, 9], [5, 7, 11], [6, 8, 12], [7, 9], [8, 10, 5], [9, 11, 15], [10, 12, 6], [11, 13, 7], [12, 14, 16], [13, 15, 17], [14, 16, 10], [15, 17, 13], [16, 18, 14], [17, 19], [18, 20, 23], [19, 21, 24], [20, 22], [21, 23], [22, 24, 19], [23, 25, 20], [24, 26, 28], [25, 27], [26, 28, 32], [27, 29, 25], [28, 30], [29, 31], [30, 32, 33], [31, 33, 27], [32, 34, 31], [33, 35, 36], [34, 36, 38], [35, 37, 34], [36, 38], [37, 39, 35], [38, 40, 42], [39, 41, 44], [40, 42, 45], [41, 43, 39], [42, 44, 46], [43, 45, 40], [44, 46, 41], [45, 47, 43], [46, 48, 49], [47, 49], [48, 0, 47]]

df = pd.read_csv('Ustar.csv')

ustar = {} 

for ind in df.index:
    ustar[(df['agent'][ind], df['prey'][ind], df['pred'][ind])] = df['utility'][ind]


def agent(ustar, clist):
    # Randomly spawing agent, predator and prey
    list_of_nodes = list(np.arange(50))
    agent1_pos = random.choice(list_of_nodes)
    agent_start = agent1_pos
    list_of_nodes.remove(agent1_pos)
    predator_pos = random.choice(list_of_nodes)
    predator_start = predator_pos
    list_of_nodes.remove(predator_pos)
    prey_pos = random.choice(list_of_nodes)
    prey_start = prey_pos

    list_of_nodes = list(np.arange(50))
    steps = 0
    if agent1_pos == prey_pos:
        win_condition = 'Win'
    elif agent1_pos == predator_pos:
        win_condition = 'Loss'
    else:
        win_condition = 'No Show'

    prey_nodes = []
    agent_nodes = []
    predator_nodes = []
    while agent1_pos != prey_pos or agent1_pos != predator_pos:
        # agents move
        neighbour = clist[agent1_pos]
        pos_states = {}
        for n in neighbour:
            u_value = ustar[(n, prey_pos, predator_pos)]
            pos_states[n] = u_value

        agent1_pos = min(pos_states, key=pos_states.get)
        agent_nodes.append(agent1_pos)
        steps += 1



        if prey_pos == agent1_pos:
            win_condition = 'Win'
            break

        # if predator_pos == agent_pos:
        #     win_condition = 'Loss'
        #     break

        # predators move
        predator_pos = predator_movement(predator_pos, agent1_pos, clist)
        predator_nodes.append(predator_pos)

        if predator_pos == agent1_pos:
            win_condition = 'Loss'
            break

        # preys move
        prey_pos = prey_movement(prey_pos, clist)
        prey_nodes.append(prey_pos)

        if prey_pos == agent1_pos:
            win_condition = 'Win'
            break

    return steps, win_condition


# steps, result, agent_start, prey_start, predator_start, prey_nodes, agent_nodes, predator_nodes = agent(ustar, connections)

# print(result)
# print(steps)
# print(agent_start)
# print(prey_start)
# print(predator_start)
# print(prey_nodes)
# print(agent_nodes)
# print(predator_nodes)



fields =  ["Win Condition", "Steps"]   
data = pd.DataFrame(columns= fields)
no_of_wins = []
ndict = {0: {'state': -1, 'connections': [49, 1, 2], 'degree': 3}, 1: {'state': -1, 'connections': [0, 2, 4], 'degree': 3}, 2: {'state': -1, 'connections': [1, 3, 0], 'degree': 3}, 3: {'state': -1, 'connections': [2, 4], 'degree': 2}, 4: {'state': -1, 'connections': [3, 5, 1], 'degree': 3}, 5: {'state': -1, 'connections': [4, 6, 9], 'degree': 3}, 6: {'state': -1, 'connections': [5, 7, 11], 'degree': 3}, 7: {'state': -1, 'connections': [6, 8, 12], 'degree': 3}, 8: {'state': -1, 'connections': [7, 9], 'degree': 2}, 9: {'state': -1, 'connections': [8, 10, 5], 'degree': 3}, 10: {'state': -1, 'connections': [9, 11, 15], 'degree': 3}, 11: {'state': -1, 'connections': [10, 12, 6], 'degree': 3}, 12: {'state': -1, 'connections': [11, 13, 7], 'degree': 3}, 13: {'state': -1, 'connections': [12, 14, 16], 'degree': 3}, 14: {'state': -1, 'connections': [13, 15, 17], 'degree': 3}, 15: {'state': -1, 'connections': [14, 16, 10], 'degree': 3}, 16: {'state': -1, 'connections': [15, 17, 13], 'degree': 3}, 17: {'state': -1, 'connections': [16, 18, 14], 'degree': 3}, 18: {'state': -1, 'connections': [17, 19], 'degree': 2}, 19: {'state': -1, 'connections': [18, 20, 23], 'degree': 3}, 20: {'state': -1, 'connections': [19, 21, 24], 'degree': 3}, 21: {'state': -1, 'connections': [20, 22], 'degree': 2}, 22: {'state': -1, 'connections': [21, 23], 'degree': 2}, 23: {'state': -1, 'connections': [22, 24, 19], 'degree': 3}, 24: {'state': -1, 'connections': [23, 25, 20], 'degree': 3}, 25: {'state': -1, 'connections': [24, 26, 28], 'degree': 3}, 26: {'state': -1, 'connections': [25, 27], 'degree': 2}, 27: {'state': -1, 'connections': [26, 28, 32], 'degree': 3}, 28: {'state': -1, 'connections': [27, 29, 25], 'degree': 3}, 29: {'state': -1, 'connections': [28, 30], 'degree': 2}, 30: {'state': -1, 'connections': [29, 31], 'degree': 2}, 31: {'state': -1, 'connections': [30, 32, 33], 'degree': 3}, 32: {'state': -1, 'connections': [31, 33, 27], 'degree': 3}, 33: {'state': -1, 'connections': [32, 34, 31], 'degree': 3}, 34: {'state': -1, 'connections': [33, 35, 36], 'degree': 3}, 35: {'state': -1, 'connections': [34, 36, 38], 'degree': 3}, 36: {'state': -1, 'connections': [35, 37, 34], 'degree': 3}, 37: {'state': -1, 'connections': [36, 38], 'degree': 2}, 38: {'state': -1, 'connections': [37, 39, 35], 'degree': 3}, 39: {'state': -1, 'connections': [38, 40, 42], 'degree': 3}, 40: {'state': -1, 'connections': [39, 41, 44], 'degree': 3}, 41: {'state': -1, 'connections': [40, 42, 45], 'degree': 3}, 42: {'state': -1, 'connections': [41, 43, 39], 'degree': 3}, 43: {'state': -1, 'connections': [42, 44, 46], 'degree': 3}, 44: {'state': -1, 'connections': [43, 45, 40], 'degree': 3}, 45: {'state': -1, 'connections': [44, 46, 41], 'degree': 3}, 46: {'state': -1, 'connections': [45, 47, 43], 'degree': 3}, 47: {'state': -1, 'connections': [46, 48], 'degree': 2}, 48: {'state': -1, 'connections': [47, 49], 'degree': 2}, 49: {'state': -1, 'connections': [48, 0], 'degree': 2}}
clist = [[49, 1, 2], [0, 2, 4], [1, 3, 0], [2, 4], [3, 5, 1], [4, 6, 9], [5, 7, 11], [6, 8, 12], [7, 9], [8, 10, 5], [9, 11, 15], [10, 12, 6], [11, 13, 7], [12, 14, 16], [13, 15, 17], [14, 16, 10], [15, 17, 13], [16, 18, 14], [17, 19], [18, 20, 23], [19, 21, 24], [20, 22], [21, 23], [22, 24, 19], [23, 25, 20], [24, 26, 28], [25, 27], [26, 28, 32], [27, 29, 25], [28, 30], [29, 31], [30, 32, 33], [31, 33, 27], [32, 34, 31], [33, 35, 36], [34, 36, 38], [35, 37, 34], [36, 38], [37, 39, 35], [38, 40, 42], [39, 41, 44], [40, 42, 45], [41, 43, 39], [42, 44, 46], [43, 45, 40], [44, 46, 41], [45, 47, 43], [46, 48], [47, 49], [48, 0]]
wins = 0
steps1 = 0

# Running 100 simulations of agnet1 in each of the 30 graphs 
for y in range(3000):
    steps, win_condition = agent(ustar, clist)
    if win_condition == 'Win':
        wins += 1
    values = pd.DataFrame([{"Win Condition" : win_condition, "Steps" : steps}])
    data = pd.concat([data, values], ignore_index=True)
no_of_wins.append(wins)
print(no_of_wins)
print('done')
data.to_csv('AgentUstar2.csv', index=False)

