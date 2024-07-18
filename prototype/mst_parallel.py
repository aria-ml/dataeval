from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional, Union

import numpy as np


class Graph:
    """
    Data structure to store graphs (based on adjacency lists)
    """

    def __init__(self, adjacency: Optional[Union[dict, Iterable, int, str]]):
        if isinstance(adjacency, (int, str)):
            self._from_vertex(adjacency)
        elif isinstance(adjacency, dict):
            self._from_dict(adjacency)
        elif isinstance(adjacency, np.ndarray) and (adjacency.shape[0] == adjacency.shape[1]):
            self._from_matrix(adjacency)
        else:
            self._adjacency = {}
        self._original_edges = self.edges

    def find_incident_edges(self, incident_to):
        vertices = incident_to.vertices if isinstance(incident_to, Graph) else Graph(incident_to).vertices
        vertices = vertices.intersection(self.vertices)
        return {e: w for v in vertices for e, w in self.edges.items() if v in e}

    def remove_redundant_edges(self, parent_graph):
        inward_vertices = self.vertices - parent_graph.vertices
        redundant_edges = [{k: v} for k, v in self.edges.items() if k.issubset(inward_vertices)]
        for redundant_edge in redundant_edges:
            self._remove_edge(redundant_edge)

    def merge_graph(self, graph):
        incident_edges = self.find_incident_edges(graph)
        for edge_v, edge_w in incident_edges.items():
            self._add_edge({edge_v: edge_w})

    @property
    def edges(self):
        """
        Returna all edges in the graph
        """
        return self._adjacency

    @property
    def vertices(self):
        return {vertex for vertices in self.edges for vertex in vertices}

    @property
    def num_vertices(self):
        return len(self.vertices)

    @staticmethod
    def is_linked_list(dictionary):
        if not isinstance(dictionary, dict):
            return False
        return any(isinstance(val, dict) for val in dictionary.values())

    def _add_edge(self, edge):
        """
        Adds an edge to the graph

        """
        head, tail, weight = self._format_edge(edge)
        vertices = frozenset([head, tail])
        self._adjacency[vertices] = min(self._adjacency.get(vertices, np.inf), weight)

    def _from_iterable(self, iterable):
        self._adjacency = {}
        try:
            n_args = len(iterable[0])
            if n_args == 2:
                self._adjacency.update({frozenset(i): None for i in iterable})
            elif n_args == 3:
                for entry in iterable:
                    if isinstance(entry[0], float):
                        weight, vertices = entry[0], entry[1:]
                    elif isinstance(entry[1], float):
                        weight = entry[1]
                        vertices = [entry[0], entry[2]]
                    else:
                        vertices, weight = entry[:2], entry[2]
                    self._adjacency.update({frozenset(vertices): weight})
        except:  # noqa: E722
            self._adjacency = {frozenset([i]): np.inf for i in iterable}

    def _from_vertex(self, vertex):
        self._adjacency = {}
        self._weight2edges = {}
        self._add_edge({vertex: np.inf})

    def _from_matrix(self, matrix):
        assert matrix == matrix.T, "initializing matrix must be square and symmetric"
        self._adjacency = {}
        self._weight2edges = {}
        triu_indices = np.triu_indices_from(matrix, k=1)
        for head, tail in triu_indices:
            weight = matrix[head, tail]
            self._add_edge({frozenset([head, tail]): weight})

    def _from_dict(self, value: dict):
        self._adjacency = {}
        self._weights2edges = {}
        if self.is_linked_list(value):
            for head in value:
                for tail, weight in value[head].items():
                    self._add_edge({frozenset([head, tail]): weight})
        else:
            for nodes, weight in value.items():
                head, tail = nodes
                self._add_edge({frozenset([head, tail]): weight})

    def _remove_edge(self, edge):
        head, tail, _ = self._format_edge(edge)
        del self._adjacency[frozenset([head, tail])]

    def _format_edge(self, edge):
        if self.is_linked_list(edge):
            head = list(edge.keys())[0]
            tail = list(edge[head].keys())[0]
            weight = edge[head][tail]
        elif isinstance(edge, dict):
            vertices = list(edge.keys())[0]
            weight = edge[vertices]
            if isinstance(vertices, (int, str)):
                head = tail = vertices
            elif len(vertices) == 2:
                head, tail = vertices
            else:
                head = tail = vertices
        elif isinstance(edge, (tuple, list, set)):
            if len(edge) == 2:
                head, tail = edge
                weight = np.inf
            elif len(edge) == 3:
                head, tail, weight = edge
                if isinstance(head, float):
                    head, weight = weight, head
                elif isinstance(tail, float):
                    tail, weight = weight, tail
            else:
                assert False, "If passing iterable, it must be length 2 or 3 (head, tail, weight)"
        else:
            raise ValueError("edge is not an instance of linked_list, dict, tuple, list or set")
        return head, tail, weight

    def __contains__(self, graph):
        if isinstance(graph, Graph):
            graph_vertices = graph.vertices
        elif isinstance(graph, set):
            graph_vertices = graph
        elif isinstance(graph, list):
            graph_vertices = set(graph)
        elif isinstance(graph, dict):
            if self.is_linked_list(graph):
                heads = set(graph)
                tails = {tail for head in graph for tail in graph[head]}
                graph_vertices = set.union(heads, tails)
            else:
                graph_vertices = {vertex for vertices in graph for vertex in vertices}
        else:
            raise ValueError("graph is not an instance of Graph, set, list or dict")
        return not self.vertices.isdisjoint(graph_vertices)

    def __str__(self):
        """
        Returns string representation of the graph
        """
        string = ""
        for nodes, weight in self._adjacency.items():
            if isinstance(nodes, frozenset) and (len(nodes) == 2):
                head, tail = nodes
                string += f"{head} -> {tail} == {weight}\n"
        return string.rstrip("\n")

    def __repr__(self):
        return f"Graph: {self._adjacency}"

    def __len__(self):
        return self.num_vertices


def merge_components(components) -> List[Graph]:
    i = 0
    while i < (len(components) - 1):
        tomerge = merge_components(components[i + 1 :])
        j = 0
        while j < len(tomerge):
            if tomerge[j] and (components[i] in tomerge[j]):
                t = tomerge.pop(j)
                components[i].merge_graph(t)
            else:
                j += 1
        components = [components[i]] + tomerge
        i += 1
    return components


def find_minimum_edge(component: Graph, graph: Graph):
    incident_edges = graph.find_incident_edges(component)
    minimum_vertices = min(incident_edges, key=incident_edges.get)  # type: ignore
    minimum_weight = incident_edges[minimum_vertices]
    minimum_edge = {minimum_vertices: minimum_weight}
    component._add_edge(minimum_edge)
    return component, minimum_edge


def prune_edges(component, graph) -> Graph:
    outer_vertices = graph.vertices - component.vertices
    for edge, weight in component.edges.items():
        if edge.isdisjoint(outer_vertices):
            component._remove_edge({edge: weight})
    return component


def minimum_spanning_tree(graph: Graph, njobs=1):
    """
    Implementation of Boruvka's algorithm
    >>> g = Graph()
    >>> g = Graph.build([0, 1, 2, 3], [[0, 1, 1], [0, 2, 1], [2, 3, 1]])
    >>> g.distinct_weight()
    >>> bg = Graph.boruvka_mst(g)
    >>> print(bg)
    1 -> 0 == 1
    2 -> 0 == 2
    0 -> 1 == 1
    0 -> 2 == 2
    3 -> 2 == 3
    2 -> 3 == 3
    """
    components: List[Graph] = [Graph(v) for v in graph.vertices]
    mst = {}
    if njobs in [0, 1]:
        while len(components) > 1:
            for i in range(len(components)):
                components[i], minimum_edge = find_minimum_edge(components[i], graph)
                mst.update(minimum_edge)
            components = merge_components(components)
            components = [prune_edges(component, graph) for component in components]
    else:
        while len(components) > 1:
            with ProcessPoolExecutor(max_workers=njobs) as pool:
                process_args = ((component, graph) for component in components)
                results = pool.map(find_minimum_edge, process_args)
                components, edges = zip(*results)
                mst.update({k: v for edge in edges for k, v in edge.items()})
                components = merge_components(components)
                components = [prune_edges(component, graph) for component in components]
    return Graph(mst)


if __name__ == "__main__":
    graph_dict = {
        0: {1: 0.6, 4: 0.4, 5: 0.9},
        1: {2: 0.8},
        2: {6: 0.7},
        3: {6: 0.3, 7: 0.9},
        4: {8: 0.7, 9: 0.3},
        5: {6: 0.5, 9: 0.8, 10: 0.2},
        6: {7: 0.5, 10: 0.9},
        7: {11: 0.4},
        9: {10: 0.4},
    }

    graph = Graph(adjacency=graph_dict)
    print(graph)
    print("\n\n=========MST SERIAL===========")
    mst_serial = minimum_spanning_tree(graph, njobs=1)
    print(mst_serial)
    print("\n\n=========MST PARALLEL=========")
    mst_parallel = minimum_spanning_tree(graph, njobs=8)
    print(mst_parallel)
