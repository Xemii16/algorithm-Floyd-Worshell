import json
import math
import random
import time
from typing import Any

INFINITY = 9_223_372_036_854_775_807


class DirectedWeightedGraph:
    def __init__(self, vertices: list, edges: set[tuple]):
        self.__edges = edges
        self.__vertices = vertices

    def add_vertice(self, vertice):
        if vertice in self.__vertices:
            raise GraphUnsupportedOperationError(f"Vertice {vertice} has already exists in graph")
        self.__vertices.append(vertice)

    def get_vertices_len(self) -> int:
        return len(self.__vertices)

    def add_vertices(self, vertices: list):
        for vertice in vertices:
            self.add_vertice(vertice)

    def remove_vertice(self, vertice):
        self.__vertices.remove(vertice)

    def get_vertices(self):
        return self.__vertices

    def get_edges_len(self):
        return len(self.__edges)

    def get_edges(self):
        return self.__edges

    def add_edge(self, weight: int, vertice_1, vertice_2):
        if not vertice_1 in self.__vertices:
            raise GraphUnsupportedOperationError(f"Graph does not have vertice: {vertice_1}")
        if not vertice_2 in self.__vertices:
            raise GraphUnsupportedOperationError(f"Graph does not have vertice: {vertice_2}")
        if vertice_1 == vertice_2:
            raise GraphUnsupportedOperationError(f"Graph is simple: first vertice is equal second vertice")
        self.__edges.add((weight, vertice_1, vertice_2))

    def add_edges(self, *edges: tuple[int, Any, Any]):
        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])

    def remove_edge(self, edge):
        self.__edges.remove(edge)

    def __find_edges(self, **param) -> list:
        start_vertice = param["start_vertice"]
        end_vertice = param["end_vertice"]
        if start_vertice is not None and end_vertice is not None:
            return list(filter(lambda edge: edge[1] == start_vertice and edge[2] == end_vertice, self.__edges))
        if start_vertice is not None:
            return list(filter(lambda edge: edge[1] == start_vertice, self.__edges))
        return list(self.__edges)

    def get_adjacency_matrix(self) -> list[list]:
        matrix = []
        for vertice_1 in self.__vertices:
            row = []
            for vertice_2 in self.__vertices:
                if vertice_1 == vertice_2:
                    row.append(0)
                    continue
                edges = self.__find_edges(start_vertice=vertice_1, end_vertice=vertice_2)
                if len(edges) == 0:
                    row.append(INFINITY)  # I think it is the biggest number :)
                    continue
                if len(edges) > 1:
                    raise GraphError(f"Edges more that 1 (current {len(edges)})")
                row.append(edges[0][0])
            matrix.append(row)
        return matrix

    @staticmethod
    def create_empty() -> DirectedWeightedGraph:
        return DirectedWeightedGraph([], set())

    @staticmethod
    def generate(n: int, density: float, min_weight=1, max_weight=100, graph=None) -> DirectedWeightedGraph:
        if graph is None:
            graph = DirectedWeightedGraph.create_empty()
            graph.add_vertices(list(range(1, n + 1)))
        c = 2
        probability = c * math.log(n) / n
        if probability >= 1:
            raise GraphError("Ohh, probability is more than 100%")
        vertices = graph.get_vertices().copy()
        random.shuffle(vertices)
        for vertice_1 in vertices:
            vertices_copy = graph.get_vertices().copy()
            random.shuffle(vertices_copy)
            for vertice_2 in vertices_copy:
                if not random.random() < probability:
                    continue
                if len(graph.__find_edges(start_vertice=vertice_1, end_vertice=vertice_2)) > 0:
                    continue
                if graph.get_edges_len() >= density * n * (n - 1):
                    break
                if vertice_1 == vertice_2:
                    continue
                graph.add_edge(random.randint(min_weight, max_weight), vertice_1, vertice_2)
                # print(f"Edge are added: ({vertice_1}, {vertice_2})")
        if graph.get_edges_len() < density * n * (n - 1):
            return DirectedWeightedGraph.generate(n, density, min_weight, max_weight, graph)
        return graph


class GraphError(Exception):
    pass


class GraphUnsupportedOperationError(Exception):
    pass


class AlgorithmFloydWorshell:

    def __init__(self, graph: DirectedWeightedGraph):
        self.graph = graph

    def work(self) -> list[list]:
        matrix = self.graph.get_adjacency_matrix()
        for k in range(0, self.graph.get_vertices_len()):
            for i in range(0, self.graph.get_vertices_len()):
                for j in range(0, self.graph.get_vertices_len()):
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
        return matrix


# this function (pretty_print_matrix) is generated by AI
def pretty_print_matrix(matrix: list[list], vertices: list):
    n = len(vertices)

    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be NxN and match vertices count")

    def fmt(x):
        return "∞" if x == INFINITY else str(x)

    cell_width = max(len(fmt(x)) for row in matrix for x in row)
    label_width = max(len(str(v)) for v in vertices)

    # column template (IMPORTANT)
    col_fmt = f"{{:>{cell_width}}}"

    # --- header ---
    print(
        " " * (label_width + 3) +
        " ".join(col_fmt.format(v) for v in vertices)
    )

    # --- rows ---
    for i, row in enumerate(matrix):
        if i == 0:
            l, r = "⎡", "⎤"
        elif i == n - 1:
            l, r = "⎣", "⎦"
        else:
            l, r = "⎢", "⎥"

        print(
            f"{vertices[i]:>{label_width}}  {l} " +
            " ".join(col_fmt.format(fmt(x)) for x in row) +
            f" {r}"
        )


# this function (measure_time) is generated by AI
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, (elapsed_time * 1000)


def save_result(n, density, min_weight, max_weight, graph_generation_time, adjacency_matrix_generation_time,
                algorithm_work_time):
    results = []
    json_path = "experiment.json"
    try:
        with open(json_path, "r") as file:
            results = json.load(file)
    except FileNotFoundError as e:
        print(f"File will be created: {e}")
    with open(json_path, "w") as file:
        results.append({
            "n": n,
            "density": density,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "graph_generation_time": graph_generation_time,
            "adjacency_matrix_generation_time": adjacency_matrix_generation_time,
            "algorithm_work_time": algorithm_work_time
        })
        json.dump(results, file)
        file.flush()


def main():
    # graph = DirectedWeightedGraph.create_empty()
    # graph.add_vertices('a', 'b', 'c', 'd', 'e')
    # graph.add_edges(
    #     (4, 'a', 'b'),
    #     (2, 'b', 'c')
    # )

    # experiment
    for _ in range(20):
        n = random.randint(20, 200) # 200
        density = random.uniform(0.90, 0.99)
        min_weight = random.randint(0, 20)
        max_weight = random.randint(80, 100)
        # n = random.randint(2, 3)
        # density = random.uniform(0.95, 0.99)
        # min_weight = random.randint(1, 4)
        # max_weight = random.randint(5, 6)
        print(f"Generating graph with n={n}, density={density}, min_weight={min_weight}, max_weight={max_weight}")
        graph, graph_generation_time = measure_time(DirectedWeightedGraph.generate, n, density, min_weight, max_weight)
        # print(f"Edges: {graph.get_edges()}")
        print(
            f"Graph with n={n}, density={density}, min_weight={min_weight}, max_weight={max_weight} generated in {graph_generation_time} ms")
        algorithm = AlgorithmFloydWorshell(graph)
        print(
            f"Graph with n={n}, density={density}, min_weight={min_weight}, max_weight={max_weight} adjacency matrix:")
        adjacency_matrix, adjacency_matrix_generation_time = measure_time(graph.get_adjacency_matrix)
        print(f"Adjacency matrix generated in {adjacency_matrix_generation_time} ms")
        pretty_print_matrix(adjacency_matrix, graph.get_vertices())
        print(
            f"Output of algorithm (graph with n={n}, density={density}, min_weight={min_weight}, max_weight={max_weight}):")
        done_matrix, algorithm_work_time = measure_time(algorithm.work)
        print(f"Algorithm has done work in {algorithm_work_time} ms")
        pretty_print_matrix(done_matrix, graph.get_vertices())
        save_result(n=n, density=density, min_weight=min_weight, max_weight=max_weight,
                    graph_generation_time=graph_generation_time,
                    adjacency_matrix_generation_time=adjacency_matrix_generation_time,
                    algorithm_work_time=algorithm_work_time)


if __name__ == "__main__":
    main()
