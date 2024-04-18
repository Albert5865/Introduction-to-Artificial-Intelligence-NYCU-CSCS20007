import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    """
        1.This code implements the A* algorithm to find the shortest path considering time constraints between two nodes in a graph. 
        2.It initializes dictionaries to store edge distances, speed limits, and the maximum speed limit. 
        3.Then, it reads edge and heuristic data from CSV files and populates the dictionaries. 
        4.The A* algorithm is performed using a priority queue to explore neighboring nodes efficiently. 
        5.It reconstructs the shortest path considering time and calculates the total time. 
        6.Finally, it returns the shortest path, total time, and the number of visited nodes.
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
            heuristic_distance[int(node)] = {
                1079387396: heuristics[0],
                1737223506: heuristics[1],
                8513026827: heuristics[2]
            }


    # Initialize variables for path, time, number of visited nodes, and previous nodes
    path = []
    time = 0
    num_visited = 0
    previous_node = {}
    # Initialize heap with start node and its heuristic time to the end node
    heap = [(heuristic_distance[start][end] / maximum_speed_limit, start)]
    
    while heap:
        num_visited += 1
        # Pop the node with the smallest time from the heap
        t, node = heapq.heappop(heap)
        # Subtract the heuristic time of the current node to the end node from the total time
        t -= heuristic_distance[node][end] / maximum_speed_limit

        # If the current node is the end node, break the loop
        if node == end:
            break
        # If the current node has no outgoing edges, skip it
        if node not in edges_distance:
            continue

        # For each destination node that the current node has an edge to
        for destination, edge_dist in edges_distance[node].items():
            # Calculate the new time
            new_time = t + edge_dist / speed_limit[node][destination] + heuristic_distance[destination][end] / maximum_speed_limit
            # If the destination node is the start node or it has been visited before with a smaller time, skip it
            if destination == start or (destination in previous_node and previous_node[destination][1] < new_time):
                continue
            # Push the destination node and its new time into the heap
            heapq.heappush(heap, (new_time, destination))
            # Record the current node as the previous node of the destination node
            previous_node[destination] = (node, new_time)

    # Start the path with the end node
    path = [end]
    # Build the path from the end node to the start node
    while path[0] in previous_node:
        # Add the time from the previous node to the current node to the total time
        time += edges_distance[previous_node[path[0]][0]][path[0]] / speed_limit[previous_node[path[0]][0]][path[0]]
        # Insert the previous node at the beginning of the path
        path.insert(0, previous_node[path[0]][0])
    
    # Return the path, the total time, and the number of visited nodes
    return path, time, num_visited
    # End your code (Part 6)

if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
