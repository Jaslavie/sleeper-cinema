# This code was reused from a previous project of mine -- Daniela
import math
from collections import deque
import random

from graph import Graph

def get_diameter(graph: Graph) -> int:
	max_distance = 0
	new_max = True
	r = random.randint(0, graph.num_nodes-1)

	while new_max:
		# Initialize BFS variables
		frontier = deque()
		frontier.append(r)
		visited = set()
		visited.add(r)
		distance = {r: 0}
		new_max = False

		while frontier:
			next_node = frontier.popleft()

			for neighbor in graph.get_neighbors(next_node):
				if neighbor not in visited:
					visited.add(neighbor)
					distance[neighbor] = distance[next_node] + 1
					frontier.append(neighbor)

		for node in distance:
			if distance[node] > max_distance:
				new_max = True
				max_distance = distance[node]
				r = node

	return max_distance


def get_clustering_coefficient(graph: Graph) -> float:
	denominator = 0

	# Sum up deg(v)(deg(v)-1)/2 for all vertices v in the graph
	for neighbor_set in graph.neighbors:
		degree = len(neighbor_set)
		denominator += degree * (degree-1) / 2

	numerator = 0
	order, neigh_v = get_degeneracy_order(graph)

	for v in order:
		for u in neigh_v[v]:
			for w in neigh_v[v]:
				if w in graph.get_neighbors(u):
					numerator += 1

	numerator /= 2 # Account for double edges from graph being undirected
	numerator *= 3

	return numerator / denominator


def get_degree_distribution(graph: Graph) -> dict[int, int]:
	degrees = {}

	for i in range(graph.get_num_nodes()):
		degree = len(graph.get_neighbors(i))

		if degree not in degrees:
			degrees[degree] = 1
		else:
			degrees[degree] += 1

	return degrees

def get_degeneracy_order(graph: Graph):
	n = graph.get_num_nodes()

	in_L = [False] * n					# Track which vertices have been moved to L already
	L = deque() 						# 1. Output array
	deg_v = [0] * n 					# 2. Number of neighbors of v not in L
	D = [set() for _ in range(n)]		# 3. List of lists of vertices v not in L with degree i
	neigh_v = [set() for _ in range(n)]	# 4. List of neighbors of v that come before v in L
	k = 0  								# 5. Initialize k

	# 2. Initialize degree tracker to each node's degree
	for i in range(n):
		deg_v[i] = len(graph.get_neighbors(i))
		D[deg_v[i]].add(i) # Step 3

	# Step 6
	for count in range(n):
		i = 0

		# Find the smallest index with non-empty buckets
		while i < n and not D[i]:
			i += 1

		# Update k for peeling
		k = max(k, i)

		v = D[i].pop()
		L.appendleft(v)
		in_L[v] = True

		# Update neighbor nodes' degrees
		for neighbor in graph.get_neighbors(v):
			if in_L[neighbor]:
				pass
			else:
				deg_v[neighbor] -= 1
				D[deg_v[neighbor] + 1].remove(neighbor)
				D[deg_v[neighbor]].add(neighbor)
				neigh_v[v].add(neighbor)

	return list(L), neigh_v
