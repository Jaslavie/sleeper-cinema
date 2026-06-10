# Some of this code was reused from a previous project of mine -- Daniela
# (Jasmine's Notes) Graph encodes the adjacency matrix (A) used to store neighbors to each node.
from collections.abc import Iterable
import pickle

class Graph:
	def __init__(self, num_nodes: int, edges: Iterable[tuple[int, int]]):
		self.num_nodes = 0
		self.neighbors = []

		# Create neighbor sets for every node
		for i in range(num_nodes):
			self.add_node()

		# Populate edges as neighbors
		for u, v in edges:
			self.add_edge(u, v)

	def get_num_nodes(self) -> int:
		return self.num_nodes

	def get_num_edges(self) -> int:
		count = 0

		# Sum degrees
		for neighbor_set in self.neighbors:
			count += len(neighbor_set)

		assert(count % 2 == 0)

		return count // 2

	def get_neighbors(self, node: int) -> Iterable[int]:
		# Return neighbors of a node or self if no neighbors
		neighbors = list(self.neighbors[node])
		return neighbors if neighbors else [node]

	def add_node(self):
		self.neighbors.append(set())
		self.num_nodes += 1

	def add_edge(self, u, v):
		self.neighbors[u].add(v)
		self.neighbors[v].add(u)

	def save_graph(self, file_name):
		with open(file_name, "wb") as file:
			pickle.dump((self.num_nodes, self.neighbors), file)

	def load_graph(self, file_name):
		with open(file_name, "rb") as file:
			self.num_nodes, self.neighbors = pickle.load(file)

	def print_neighbor_sets(self):
		for node_index in range(len(self.neighbors)):
			if self.get_neighbors(node_index):
				print(f"{node_index}: {self.get_neighbors(node_index)}")

	def edge_exists(self, u, v):
		return v in self.neighbors[u]