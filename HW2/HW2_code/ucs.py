import csv
import heapq
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    """
        1.The code initializes dictionaries for edges (edges) and heuristics (heuristic), and a variable to store the maximum speed (max_speed).
        2.It reads data from the CSV files 'edges.csv' and 'heuristic.csv', populating the dictionaries accordingly.
        3.Variables for path, distance, and visited nodes are initialized.
        4.UCS algorithm is performed using heapq for priority queue implementation to find the shortest path.
        5.The shortest path is reconstructed, and the total distance calculated.
        6.The function returns the shortest path, total distance, and number of visited nodes.
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
    dist = float(0)
    num_visited = int(0)
    previous_node = {}
    heap = []
    heapq.heappush(heap, (float(0), start))
    
    while heap:
        num_visited += 1
        distance, node = heapq.heappop(heap)

        if (node == end):
            break
        if node not in edges_distance:
            continue
        for destination in edges_distance[node]:
            if destination == start or destination in previous_node and previous_node[destination][1] < distance + edges_distance[node][destination]:
                continue
            heapq.heappush(heap, (distance + edges_distance[node][destination], destination))
            previous_node[destination] = (node, distance + edges_distance[node][destination])

    path.append(end)
    while path[0] in previous_node:
        dist += edges_distance[previous_node[path[0]][0]][path[0]]
        path.insert(0, previous_node[path[0]][0])
    
    return path, dist, num_visited
    # End your code (Part 3)



if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
