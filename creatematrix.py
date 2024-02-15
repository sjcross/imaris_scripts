from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import numpy as np

# graph1 = csr_matrix([[0,1,0,0,0,0],
#                     [1,0,3,0,0,0],
#                     [0,3,0,5,6,4],
#                     [0,0,5,0,8,0],
#                     [0,0,6,8,0,3],
#                     [0,0,4,0,3,0]])
# print(graph1)
# dist_mat1 = dijkstra(csgraph=graph1, directed=False)
# print(dist_mat1)

# graph2 = csr_matrix([[0,1,0,0,0,0],
#                     [0,0,3,0,0,0],
#                     [0,0,0,5,6,4],
#                     [0,0,0,0,8,0],
#                     [0,0,0,0,0,3],
#                     [0,0,0,0,0,0]])
# print(graph2)
# dist_mat2 = dijkstra(csgraph=graph2, directed=False)
# print(dist_mat2)

row = np.array([0,1,2,2,2,3,4])
col = np.array([1,2,3,4,5,4,5])
data = np.array([1,3,5,6,4,8,3])
graph3 = csr_matrix((data, (row, col)),shape=(6,6))
print(graph3)
dist_mat3, predecessors = dijkstra(csgraph=graph3, directed=False, return_predecessors=True)
print(dist_mat3)
print(predecessors)

start_pt = 0
end_pt = 3
predecessor = start_pt
path = [predecessor]
print(predecessor)
while predecessor != end_pt:
    predecessor = predecessors[end_pt,predecessor]
    path.append(predecessor)

print(path)