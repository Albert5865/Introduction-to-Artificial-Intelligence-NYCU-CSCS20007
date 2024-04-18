import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    """
        1.This code implements the A* algorithm to find the shortest path between two nodes in a graph. 
        2.It initializes dictionaries to store edge distances and heuristic estimates. 
        3.Performs the A* algorithm by iteratively exploring the nodes with the lowest combined cost of distance 
          and heuristic estimate until the destination node is reached. 
        4.The algorithm reconstructs the shortest path and calculates the total distance traveled. 
        5.Finally, it returns the shortest path, total distance, and number of visited nodes.
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
            

    # Initialize variables for path, distance, number of visited nodes, and previous nodes
    path = []
    dist = 0
    num_visited = 0
    previous_node = {}
    # Initialize heap with start node and its heuristic distance to the end node
    heap = [(heuristic_distance[start][end], start)]
    
    while heap:
        num_visited += 1
        # Pop the node with the smallest distance from the heap
        distance, node = heapq.heappop(heap)
        # Subtract the heuristic distance of the current node to the end node from the total distance
        distance -= heuristic_distance[node][end]

        # If the current node is the end node, break the loop
        if node == end:
            break
        # If the current node has no outgoing edges, skip it
        if node not in edges_distance:
            continue

        # For each destination node that the current node has an edge to
        for destination, edge_dist in edges_distance[node].items():
            # Calculate the new distance
            new_dist = distance + edge_dist + heuristic_distance[destination][end]
            # If the destination node is the start node or it has been visited before with a smaller distance, skip it
            if destination == start or (destination in previous_node and previous_node[destination][1] < new_dist):
                continue
            # Push the destination node and its new distance into the heap
            heapq.heappush(heap, (new_dist, destination))
            # Record the current node as the previous node of the destination node
            previous_node[destination] = (node, new_dist)

    # Start the path with the end node
    path = [end]
    # Build the path from the end node to the start node
    while path[0] in previous_node:
        # Add the distance from the previous node to the current node to the total distance
        dist += edges_distance[previous_node[path[0]][0]][path[0]]
        # Insert the previous node at the beginning of the path
        path.insert(0, previous_node[path[0]][0])
    
    # Return the path, the total distance, and the number of visited nodes
    return path, dist, num_visited
    # End your code (Part 4)



if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
