import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    """
        1.The code begins by initializing dictionaries to store edge distances (edges_distance), 
          speed limits (speed_limit), and the maximum speed limit (maximum_speed_limit).
        2.It then reads data from the CSV files 'edges.csv' and 'heuristic.csv', 
          populating the dictionaries with edge distances and heuristic distances respectively.
        3.Next, the code initializes variables for the path, distance, number of visited nodes, 
          and a dictionary to store previous nodes in the BFS traversal.
        4.The breadth-first search algorithm is performed to find the shortest path from the start node to the end node.
        5.Finally, the code reconstructs the shortest path and calculates the total distance along that path, 
          returning the path, distance, and number of visited nodes as a tuple.
    """
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
    bfs_queue = [start]
    
    while bfs_queue:
        num_visited += 1
        node = bfs_queue.pop(0)
        if node == end:
            break
        if node not in edges_distance:
            continue
        for destination in edges_distance[node]:
            if destination == start or destination in previous_node:
                continue
            bfs_queue.append(destination)
            previous_node[destination] = node

    path.append(end)
    while path[0] in previous_node:
        dist += edges_distance[previous_node[path[0]]][path[0]]
        path.insert(0, previous_node[path[0]])
    
    return path, dist, num_visited
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
