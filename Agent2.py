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

# Method that decides predator movement
def prey_movement(prey_position, list1):
    nlist = list1[prey_position]
    new_pos = random.choice(nlist)
    return new_pos

# Agent method that runs away from the prey and tries to catch the prey with the logic that gives better result than the given logic
def agent2(ndict, clist):
    # Randomly spawing agent, predator and prey
    list_of_nodes = list(np.arange(50))
    agent_pos = random.choice(list_of_nodes)
    list_of_nodes.remove(agent_pos)
    ndict[agent_pos]['state'] = 1
    predator_pos = random.choice(list_of_nodes)
    list_of_nodes.remove(predator_pos)
    ndict[predator_pos]['state'] = -10
    prey_pos = random.choice(list_of_nodes)
    ndict[prey_pos]['state'] = 10

    list_of_nodes = list(np.arange(50))
    steps = 0
    win_condition = 'out of steps'
    while steps < 5000:
        # agents move
        ndict[agent_pos]['state'] = -1
        ndict[predator_pos]['state'] = -1
        ndict[prey_pos]['state'] = -1
        curr_dist_prey = dijkstra(agent_pos, prey_pos, clist)
        curr_dist_predator = dijkstra(agent_pos, predator_pos, clist)
        neighbours = clist[agent_pos]
        priority = list(np.zeros(len(neighbours)))
        for x in range(len(neighbours)):
            n_dist_prey = dijkstra(neighbours[x], prey_pos, clist)
            n_dist_predator = dijkstra(neighbours[x], predator_pos, clist)
            if predator_pos in n_dist_prey:
                priority[x] = 6
            if len(curr_dist_prey) > len(n_dist_prey) and len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 1
            elif len(curr_dist_prey) == len(n_dist_prey) and len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 2
            elif len(curr_dist_prey) < len(n_dist_prey) and len(curr_dist_predator) < len(n_dist_predator):
                priority[x] = 3
            elif len(curr_dist_predator) == len(n_dist_predator):
                priority[x] = 4
            else:
                priority[x] = 5
        priority_max = min(priority)
        move_to_list = []
        for y in range(len(priority)):
            if priority[y] == priority_max:
                move_to_list.append(neighbours[y])
        if priority_max == 5:
            agent_pos = agent_pos
        else:
            agent_pos = random.choice(move_to_list)

        
        ndict[agent_pos]['state'] = 1
        steps += 1



        if prey_pos == agent_pos:
            win_condition = 'Win'
            break

        if predator_pos == agent_pos:
            win_condition = 'Loss'
            break

        # predators move
        pred_new_pos = predator_movement(predator_pos, agent_pos, clist)
        predator_pos = pred_new_pos
        ndict[predator_pos]['state'] = -10

        if predator_pos == agent_pos:
            win_condition = 'Loss'
            break

        # preys move
        prey_pos = prey_movement(prey_pos, clist)
        ndict[prey_pos]['state'] = 10

        if prey_pos == agent_pos:
            win_condition = 'Win'
            break

    return steps, win_condition, ndict

fields =  ["Win Condition", "Steps"]   
data = pd.DataFrame(columns= fields)
graph = 0
no_of_wins = []
ndict = {0: {'state': -1, 'connections': [49, 1, 2], 'degree': 3}, 1: {'state': -1, 'connections': [0, 2, 4], 'degree': 3}, 2: {'state': -1, 'connections': [1, 3, 0], 'degree': 3}, 3: {'state': -1, 'connections': [2, 4], 'degree': 2}, 4: {'state': -1, 'connections': [3, 5, 1], 'degree': 3}, 5: {'state': -1, 'connections': [4, 6, 9], 'degree': 3}, 6: {'state': -1, 'connections': [5, 7, 11], 'degree': 3}, 7: {'state': -1, 'connections': [6, 8, 12], 'degree': 3}, 8: {'state': -1, 'connections': [7, 9], 'degree': 2}, 9: {'state': -1, 'connections': [8, 10, 5], 'degree': 3}, 10: {'state': -1, 'connections': [9, 11, 15], 'degree': 3}, 11: {'state': -1, 'connections': [10, 12, 6], 'degree': 3}, 12: {'state': -1, 'connections': [11, 13, 7], 'degree': 3}, 13: {'state': -1, 'connections': [12, 14, 16], 'degree': 3}, 14: {'state': -1, 'connections': [13, 15, 17], 'degree': 3}, 15: {'state': -1, 'connections': [14, 16, 10], 'degree': 3}, 16: {'state': -1, 'connections': [15, 17, 13], 'degree': 3}, 17: {'state': -1, 'connections': [16, 18, 14], 'degree': 3}, 18: {'state': -1, 'connections': [17, 19], 'degree': 2}, 19: {'state': -1, 'connections': [18, 20, 23], 'degree': 3}, 20: {'state': -1, 'connections': [19, 21, 24], 'degree': 3}, 21: {'state': -1, 'connections': [20, 22], 'degree': 2}, 22: {'state': -1, 'connections': [21, 23], 'degree': 2}, 23: {'state': -1, 'connections': [22, 24, 19], 'degree': 3}, 24: {'state': -1, 'connections': [23, 25, 20], 'degree': 3}, 25: {'state': -1, 'connections': [24, 26, 28], 'degree': 3}, 26: {'state': -1, 'connections': [25, 27], 'degree': 2}, 27: {'state': -1, 'connections': [26, 28, 32], 'degree': 3}, 28: {'state': -1, 'connections': [27, 29, 25], 'degree': 3}, 29: {'state': -1, 'connections': [28, 30], 'degree': 2}, 30: {'state': -1, 'connections': [29, 31], 'degree': 2}, 31: {'state': -1, 'connections': [30, 32, 33], 'degree': 3}, 32: {'state': -1, 'connections': [31, 33, 27], 'degree': 3}, 33: {'state': -1, 'connections': [32, 34, 31], 'degree': 3}, 34: {'state': -1, 'connections': [33, 35, 36], 'degree': 3}, 35: {'state': -1, 'connections': [34, 36, 38], 'degree': 3}, 36: {'state': -1, 'connections': [35, 37, 34], 'degree': 3}, 37: {'state': -1, 'connections': [36, 38], 'degree': 2}, 38: {'state': -1, 'connections': [37, 39, 35], 'degree': 3}, 39: {'state': -1, 'connections': [38, 40, 42], 'degree': 3}, 40: {'state': -1, 'connections': [39, 41, 44], 'degree': 3}, 41: {'state': -1, 'connections': [40, 42, 45], 'degree': 3}, 42: {'state': -1, 'connections': [41, 43, 39], 'degree': 3}, 43: {'state': -1, 'connections': [42, 44, 46], 'degree': 3}, 44: {'state': -1, 'connections': [43, 45, 40], 'degree': 3}, 45: {'state': -1, 'connections': [44, 46, 41], 'degree': 3}, 46: {'state': -1, 'connections': [45, 47, 43], 'degree': 3}, 47: {'state': -1, 'connections': [46, 48], 'degree': 2}, 48: {'state': -1, 'connections': [47, 49], 'degree': 2}, 49: {'state': -1, 'connections': [48, 0], 'degree': 2}}
clist = [[49, 1, 2], [0, 2, 4], [1, 3, 0], [2, 4], [3, 5, 1], [4, 6, 9], [5, 7, 11], [6, 8, 12], [7, 9], [8, 10, 5], [9, 11, 15], [10, 12, 6], [11, 13, 7], [12, 14, 16], [13, 15, 17], [14, 16, 10], [15, 17, 13], [16, 18, 14], [17, 19], [18, 20, 23], [19, 21, 24], [20, 22], [21, 23], [22, 24, 19], [23, 25, 20], [24, 26, 28], [25, 27], [26, 28, 32], [27, 29, 25], [28, 30], [29, 31], [30, 32, 33], [31, 33, 27], [32, 34, 31], [33, 35, 36], [34, 36, 38], [35, 37, 34], [36, 38], [37, 39, 35], [38, 40, 42], [39, 41, 44], [40, 42, 45], [41, 43, 39], [42, 44, 46], [43, 45, 40], [44, 46, 41], [45, 47, 43], [46, 48, 49], [47, 49], [48, 0, 47]]
wins = 0
steps1 = 0

# Running 100 simulations of agnet1 in each of the 30 graphs 
for y in range(3000):
    steps, win_condition, ndict = agent2(ndict, clist)
    if win_condition == 'Win':
        wins += 1
    values = pd.DataFrame([{"Win Condition" : win_condition, "Steps" : steps}])
    data = pd.concat([data, values], ignore_index=True)
no_of_wins.append(wins)
print(no_of_wins)
print('done')
data.to_csv('AgentV.csv', index=False)

            
