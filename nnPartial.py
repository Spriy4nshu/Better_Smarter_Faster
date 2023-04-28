import numpy as np
import pandas as pd
from queue import PriorityQueue

#Dijkstra returns the shortest path between 2 nodes
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

# activation function
def sigmoid(x, deriv = False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))



connections = [[49, 1, 2], [0, 2, 4], [1, 3, 0], [2, 4], [3, 5, 1], [4, 6, 9], [5, 7, 11], [6, 8, 12], [7, 9], [8, 10, 5], [9, 11, 15], [10, 12, 6], [11, 13, 7], [12, 14, 16], [13, 15, 17], [14, 16, 10], [15, 17, 13], [16, 18, 14], [17, 19], [18, 20, 23], [19, 21, 24], [20, 22], [21, 23], [22, 24, 19], [23, 25, 20], [24, 26, 28], [25, 27], [26, 28, 32], [27, 29, 25], [28, 30], [29, 31], [30, 32, 33], [31, 33, 27], [32, 34, 31], [33, 35, 36], [34, 36, 38], [35, 37, 34], [36, 38], [37, 39, 35], [38, 40, 42], [39, 41, 44], [40, 42, 45], [41, 43, 39], [42, 44, 46], [43, 45, 40], [44, 46, 41], [45, 47, 43], [46, 48, 49], [47, 49], [48, 0, 47]]


df = pd.read_csv('Ustar.csv')

ustar = {} 

for ind in df.index:
    ustar[(df['agent'][ind], df['prey'][ind], df['pred'][ind])] = df['utility'][ind]

output_scale=15

map={}

# giving inputs and labels
t_inputs=[]
t_outputs=[]
for x in range(50):
    for y in range(50):
        for z in range(50):
            state = (x, y, z)
            if ustar[state] < 10:
                t_outputs.append([ustar[state]/output_scale]) 
                dist_pred = len(dijkstra(x, z, connections))
                agent_pos = x
                predator_pos = z
                t_inputs.append([dist_pred, agent_pos, predator_pos])
            else:
                map[state] = ustar[state]



train_outputs = np.array(t_outputs)
train_inputs = np.array(t_inputs, dtype=float)

# np.random.seed(1)
weight1 = 1 * np.random.randn(3, 6) #(input size*hiddenlayer1 size)
weight12 = 1 * np.random.randn(6, 6) #(hiddenlayer1 size*hiddenlayer2 size)
weight23 = 1 * np.random.randn(6, 6) #(hiddenlayer1 size*hiddenlayer2 size)
weight3 = 1 * np.random.randn(6, 1) #(hiddenlayer2 size*output size)

err_mean=10
while err_mean>0.06:
    ####FORWARD PROPAGATION
    l = np.dot(train_inputs, weight1) #dot product of X (input) and first set of weights (3x2)
    l2 = sigmoid(l) #activation function
    l3 = np.dot(l2, weight12) #dot product of hidden layer (z2) and second set of weights (3x1)
    l4 = sigmoid(l3) #activation function
    l5 = np.dot(l4, weight23)
    l6 = sigmoid(l5)
    l7 = np.dot(l6, weight3)
    output = sigmoid(l7)

    ####BACKWARD PROPAGATION
    output_error = train_outputs - output # error in output
    output_delta = output_error * sigmoid(output, deriv= True)
    
    l2_error = output_delta.dot(weight3.T) #z2 error: how much our hidden2 layer weights contribute to output error
    l2_delta = l2_error * sigmoid(l2, deriv= True) #applying derivative of sigmoid to z2 error
    
    l4_error = l2_delta.dot(weight12.T) #z2 error: how much our hidden1 layer weights contribute to output error
    l4_delta = l4_error * sigmoid(l4, deriv= True) #applying derivative of sigmoid to z2 error
    
    l6_error = l4_delta.dot(weight23.T) #z2 error: how much our hidden1 layer weights contribute to output error
    l6_delta = l6_error * sigmoid(l6, deriv= True)

    weight1 += 0.01*train_inputs.T.dot(l6_delta) # adjusting first set (input -> hidden) weights
    weight12 += 0.01*l2.T.dot(l2_delta) # adjusting second set (hidden1 -> hidden2) weights
    weight23 += 0.01*l4.T.dot(l4_delta) # adjusting second set (hidden2 -> output) weights
    weight3 += 0.01*l6.T.dot(output_delta)


    err_mean =np.mean(np.square(train_outputs - output))
    print("Loss: " + str(err_mean))


print(weight1)
print(weight12)
print(weight23)
print(weight3)
print(output)


## finding the ustar value:
# inputs=[]
# for x in range(50):
#     for y in range(50):
#         for z in range(50):
#             state = (x, y, z)
#             if ustar[state] < 10:
#                 dist_prey = len(dijkstra(x, y, connections))
#                 dist_pred = len(dijkstra(x, z, connections))
#                 dist = len(dijkstra(y, z, connections))
#                 inputs.append([dist_prey, dist_pred, dist])
#             else:
#                 output = ustar[state]
# l = np.dot(inputs, weight1) #dot product of X (input) and first set of weights (3x2)
# l2 = sigmoid(l) #activation function
# l3 = np.dot(l2, weight12) #dot product of hidden layer (z2) and second set of weights (3x1)
# l4 = sigmoid(l3) #activation function
# l5 = np.dot(l4, weight23)
# l6 = sigmoid(l5)
# l7 = np.dot(l6, weight3)
# output = sigmoid(l7)

# print(output)