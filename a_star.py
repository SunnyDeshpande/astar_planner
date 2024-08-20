import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(0)

class Graph():
    def __init__(self):
        self.graph = {}
        self.occupancy_costs = {}
    
    def add_node(self, node, cost=0):
        if node not in self.graph:
            self.graph[node] = []
            self.occupancy_costs[node] = cost
        else:
            #print("Node already in graph!")
            pass

    def add_edge(self, from_node, to_node, cost=1, from_cost=1, to_cost=1):
        self.add_node(from_node,from_cost)
        self.add_node(to_node,to_cost)  
        self.graph[from_node].append((to_node, to_cost-from_cost))

    def remove_node(self, del_node):
        if del_node in self.graph:
            del self.graph[del_node]
            for node, neighbours in self.graph.items():
                self.graph[node] = [(neigh, cost) for neigh, cost in neighbours if neigh != del_node]
        else:
            print("Node not in graph!")

    def remove_edge(self, from_node, to_node):
        self.graph[from_node] = [(neigh, cost) for neigh, cost in self.graph[from_node] if neigh != to_node]

    def update_cost(self, from_node, to_node, new_cost):
        self.remove_edge(from_node, to_node)
        self.add_edge(from_node, to_node, new_cost)

    def get_neighbours(self, node):
        return self.graph.get(node, [])

    def get_occupancy_cost(self, node):
        return self.occupancy_costs[node]
    
    def get_graph(self):
        return self.graph
    
    def display_graph(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")

class ASTAR():
    def __init__(self, input_graph, heuristic_factor=1):
        if input_graph:
            self.graph = input_graph
        else:    
            self.graph = Graph()
        self.path = []
        self.g_costs = {}
        self.f_costs = {}
        self.occupancy_costs = {}
        self.heuristic_costs = {}
        self.heuristic_factor = heuristic_factor

    def heuristic(self, node, goal):
        return self.heuristic_factor * np.sqrt(abs(node[0] - goal[0])**2 + abs(node[1] - goal[1])**2)
    
    def search(self, start, goal): 
        queue = [(0, start)]
        visited = set()
        g_costs = {start: 0}
        f_costs = {start: self.heuristic(start, goal)}
        occupancy_costs = {}
        heuristic_costs = {}
        previous = {start: None}

        while queue:
            current_cost, current_node = heapq.heappop(queue)

            if current_node not in visited:
                visited.add(current_node)

            if current_node == goal:
                print("Current node is goal!")
                break

            for neighbour, weight in self.graph.get_neighbours(current_node):
                # cost of getting to node, through previous ones
                # f(n) = g(n) + h(n)
                occupancy_cost = self.graph.get_occupancy_cost(neighbour)
                tent_g_cost = g_costs[current_node] + weight + occupancy_cost
                tent_f_cost = tent_g_cost + self.heuristic(neighbour, goal)
                
                if neighbour not in f_costs or tent_f_cost < f_costs[neighbour]:
                    occupancy_costs[neighbour] = occupancy_cost
                    g_costs[neighbour] = tent_g_cost
                    f_costs[neighbour] = tent_f_cost
                    heuristic_costs[neighbour] = self.heuristic(neighbour, goal)
                    previous[neighbour] = current_node
                    heapq.heappush(queue, (tent_f_cost, neighbour))
        else:
            print("No path found.")
            self.path = []
            self.g_costs = {}
            self.f_costs = {}
            self.occupancy_costs = {}
            self.heuristic_costs = {}
            return

        temp_path = []

        # while there exists a path that leads to goal
        while previous[goal] is not None: 
            # iterate backwards
            temp_path.insert(0, goal)
            goal = previous[goal]
        temp_path.insert(0, start)

        self.path = temp_path
        self.g_costs = g_costs
        self.f_costs = f_costs
        self.occupancy_costs = occupancy_costs
        self.heuristic_costs = heuristic_costs
    
    def get_astar_path(self, start, goal):
        self.search(start, goal)
        return self.path, self.g_costs, self.f_costs, self.occupancy_costs, self.heuristic_costs
    
class OccupancyGrid:
    def __init__(self, size=(32,32), obstacle_density=0.1, obstacle_size=1, penalty_factor=2, liberal=(0,0), conservative=(1,1)):
        self.size = size
        self.grid = np.zeros(size, dtype=float)
        self.obstacle_size = obstacle_size
        self.penalty_factor = penalty_factor
        self.generate_random_obstacles(obstacle_density)
        self.liberal = liberal
        self.conservative = conservative

    def generate_random_obstacles(self, obstacle_density):
        num_obstacles = int(self.size[0] * self.size[1] * obstacle_density)
        obstacle_indices = np.random.choice(self.size[0] * self.size[1], num_obstacles)
        for index in obstacle_indices:
            row = index // self.size[1]
            col = index % self.size[1]
            self.add_obstacles(row,col)

    def add_obstacles(self, row, col):
        for i in range(-self.obstacle_size, self.obstacle_size + 1):
            for j in range(-self.obstacle_size, self.obstacle_size + 1):
                new_row = row + i
                new_col = col + j
                if 0 <= new_row < self.size[0] and 0 <= new_col < self.size[1]:
                    if(i == 0 and j == 0):
                        self.grid[new_row, new_col] = np.max([self.grid[new_row, new_col], 100.0*self.penalty_factor])
                    else:
                        self.grid[new_row, new_col] =  np.max([self.grid[new_row, new_col], (100.0- (np.max([abs(i),abs(j)])/self.obstacle_size)*40.0)*self.penalty_factor])

    def visualize(self, path=None, start=None, goal=None):
        cmap = mcolors.ListedColormap(['white', 'green' ,'orange', 'red', 'black'])
        bounds = [0.0, 35.0, 70.0, 85.0, 99.0, 100.0]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap=cmap, norm=norm)

        
        plt.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(0, self.grid.shape[1], 1))
        plt.yticks(np.arange(0, self.grid.shape[0], 1))
        plt.gca().set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
    
        if path:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], color='blue', linewidth=2, marker='o', markersize=4)
            plt.plot(self.liberal[1], self.liberal[0], color='brown', marker='o', markersize=4)
            plt.plot(self.conservative[1], self.conservative[0], color='purple', marker='o', markersize=4)
        
        if start:
            plt.plot(start[1], start[0], color='green', marker='s', markersize=10, label='Start')
        if goal:
            plt.plot(goal[1], goal[0], color='red', marker='X', markersize=10, label='Goal')
    
        plt.title('Occupancy Grid with Larger Obstacles and Partial Obstacles')
        plt.legend()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

def occupancy_grid_to_graph(occupancy_grid, cost_scaling_factor=1.1):
    graph = Graph()
    rows = len(occupancy_grid)
    cols = len(occupancy_grid[0])
    cost_scaling = cost_scaling_factor * np.max([rows, cols])

    for row in range(rows):
        for col in range(cols):
            if occupancy_grid[row][col] < 100.0:
                current_node = (row, col)

                for d_row, d_col in [(-1,0), (0,-1), (1,0), (0,1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    neighbour_row, neighbour_col = row + d_row, col + d_col
                    if 0 <= neighbour_row < rows and 0 <= neighbour_col < cols:
                        if occupancy_grid[neighbour_row][neighbour_col] < 100.0:
                            # can be modified to add actual coordinates
                            neighbour_node = (neighbour_row, neighbour_col)
                            graph.add_edge(current_node, neighbour_node, 1, occupancy_grid[row][col], occupancy_grid[row+d_row][col+d_col])
    
    return graph

#liberal = (35, 27)
#conservative = (35, 26)

liberal = (21, 18)
conservative = (21, 19)
scale = 50

print("starting.")
occupancy_grid = OccupancyGrid(size=(scale,scale), obstacle_density=0.007, obstacle_size=5, penalty_factor=1.05, liberal=liberal, conservative=conservative)
print("made grid.")
graph = occupancy_grid_to_graph(occupancy_grid.grid, cost_scaling_factor=1.1)
print("made graph.")
planner = ASTAR(graph, 40)
print("made planner.")
path, g_costs, f_costs, occupancy_costs, heuristic_costs = planner.get_astar_path((0,0), (scale-1,scale-1))

occupancy_grid.visualize(path)


#print("liberal(brown): " + str(liberal))
#print("g cost: " + str(g_costs[liberal]))
#print("occupancy cost: " + str(occupancy_costs[liberal]))
#print("heuristic cost: " + str(heuristic_costs[liberal]))
#print("f cost: " + str(f_costs[liberal]))

#print("\nconservative(purple): " + str(conservative))
#print("g cost: " + str(g_costs[conservative]))
#print("occupancy cost: " + str(occupancy_costs[conservative]))
#print("heuristic cost: " + str(heuristic_costs[conservative]))
#print("f cost: " + str(f_costs[conservative]))

#print("\n")
#for neighbour in graph.get_neighbours(liberal):
#    print(neighbour)

print("\ndone.")