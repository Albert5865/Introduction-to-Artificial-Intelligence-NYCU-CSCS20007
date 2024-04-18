import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
        1.The code initializes dictionaries to store edge distances (edges_distance), 
          speed limits (speed_limit), and maximum speed (max_speed).
        2.It reads data from the CSV files 'edges.csv' and 'heuristic.csv', populating the dictionaries.
        3.Variables for path, distance, number of visited nodes, 
          and a dictionary to store previous nodes in the DFS traversal are initialized.
        4.Depth-first search algorithm is performed to find a path from the start node to the end node.
        5.The path is reconstructed, and the total distance calculated.
        6.The function returns the path, distance, and number of visited nodes.
    """
    # Initialize dictionaries to store edge distances, speed limits, and maximum speed limit
    edges_distance = {}
    speed_limit = {}
    maximum_speed_limit = float('-inf')

    # Read data from the CSV files 'edges.csv' and 'heuristic.csv'
    with open('edges.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            source, destination, distance, speed = map(float, row)  # Change int to float
            edges_distance.setdefault(int(source), {})[int(destination)] = distance
            speed_limit.setdefault(int(source), {})[int(destination)] = speed / 3.6
            maximum_speed_limit = max(maximum_speed_limit, speed / 3.6)

    heuristic_distance = {}
    with open('heuristic.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            node, *heuristics = map(float, row)
            heuristic_distance[int(node)] = heuristics

    # Initialize variables for path, distance, number of visited nodes, and previous nodes
    path = []
    dist = 0.0
    num_visited = 0
    previous_node = {}
    dfs_stack = []
    dfs_stack.append(start)
    
    # Perform depth-first search to find a path from start to end
    while dfs_stack:
        num_visited += 1
        node = dfs_stack.pop()
        # If the end node is reached, exit the loop
        if node == end: break
        # Check if the current node has outgoing edges
        for dest in edges_distance.get(node, {}):
            if dest == start or dest in previous_node: continue
            dfs_stack.append(dest)
            previous_node[dest] = node

    # Reconstruct the path and calculate the total distance
    path.append(end)
    while path[0] in previous_node:
        dist += edges_distance[previous_node[path[0]]][path[0]]
        path.insert(0, previous_node[path[0]])
    
    # Return the path, distance, and number of visited nodes
    return path, dist, num_visited
    # End your code (Part 2)
    


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
