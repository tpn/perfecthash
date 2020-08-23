#===============================================================================
# Imports
#===============================================================================
import sys

from .util import (
    round_up_power_of_2,
    round_up_next_power_of_2,
)

from perfecthash._graph import (
    hash_all,
    hash_key,
)

#===============================================================================
# Globals
#===============================================================================

#===============================================================================
# Helper Methods
#===============================================================================

#===============================================================================
# Classes
#===============================================================================

def is_empty(v):
    return v == -1

class GraphIterator:
    __slots__ = [
        'vertex',
        'edge',
        'graph',
        'depth',
        'indent',
    ]
    def __init__(self, graph, vertex, depth):
        self.graph = graph
        self.vertex = vertex
        self.edge = graph.first[vertex]
        self.depth = depth
        self.indent = '    ' * depth

    def next_neighbor(self):
        if is_empty(self.edge):
            return

        graph = self.graph
        vertex = graph.edges[self.edge]
        if vertex == self.vertex:
            neighbor = graph.edges[self.edge + graph.num_edges]
        else:
            neighbor = vertex

        self.edge = graph.next[self.edge]
        return neighbor


class Graph:
    __slots__ = [
        'out',
        'keys',
        'next',
        'first',
        'seed1',
        'seed2',
        'edges',
        'pairs',
        'hashed',
        'visited',
        'assigned',
        'num_keys',
        'num_edges',
        'edge_mask',
        'hash_mask',
        'vertices1',
        'vertices2',
        'collisions',
        'index_mask',
        'seed3_byte1',
        'seed3_byte2',
        'vertex_mask',
        'index_bitmap',
        'num_vertices',
        'traversal_depth',
        'total_traversals',
        'max_traversal_depth',
    ]
    def __init__(self, keys, out=None):
        import numpy as np
        self.out = out
        if not self.out:
            self.out = lambda s: sys.stdout.write(f'{s}\n')
        self.num_keys = num_keys = len(keys)
        num_edges = round_up_power_of_2(num_keys)
        num_vertices = round_up_next_power_of_2(num_edges)
        total_edges = num_edges * 2
        self.keys = keys
        self.hashed = False
        self.num_edges = num_edges
        self.num_vertices = num_vertices
        self.first = np.full(num_vertices, -1, dtype=np.int32)
        self.edges = np.full(total_edges, -1, dtype=np.int32)
        self.next = np.full(total_edges, -1, dtype=np.int32)
        self.pairs = np.zeros(num_keys * 2, dtype=np.uint32)
        self.assigned = np.zeros(num_vertices, dtype=np.uint32)
        self.vertices1 = np.zeros(num_keys, dtype=np.uint32)
        self.vertices2 = np.zeros(num_keys, dtype=np.uint32)
        self.edge_mask = num_edges - 1
        self.index_mask = num_edges - 1
        self.hash_mask = num_vertices - 1
        self.vertex_mask = num_vertices - 1
        self.visited = set()
        self.collisions = 0
        self.index_bitmap = set()
        self.traversal_depth = 0
        self.total_traversals = 0
        self.max_traversal_depth = 0

    def hash(self, seed1, seed2, seed3_byte1, seed3_byte2):
        hash_all(
            self.num_keys,
            self.num_edges,
            self.hash_mask,
            seed1,
            seed2,
            seed3_byte1,
            seed3_byte2,
            self.keys,
            self.vertices1,
            self.vertices2,
            self.first,
            self.next,
            self.edges,
            self.pairs
        )
        self.hashed = True
        self.seed1 = seed1
        self.seed2 = seed2
        self.seed3_byte1 = seed3_byte1
        self.seed3_byte2 = seed3_byte2

    def hash_key(self, key):
        return hash_key(
            key,
            self.hash_mask,
            self.seed1,
            self.seed2,
            self.seed3_byte1,
            self.seed3_byte2,
        )

    def edge_id(self, vertex1, vertex2):
        edge = self.first[vertex1]
        assert not is_empty(edge), (edge, vertex1, vertex2)

        if self.check_edge(edge, vertex1, vertex2):
            edge_id = edge
        else:
            count = 0
            while True:
                count += 1
                edge = self.next[edge]
                assert not is_empty(edge), (edge, vertex1, vertex2)
                if self.check_edge(edge, vertex1, vertex2):
                    edge_id = edge
                    break

        return edge_id

    def check_edge(self, edge, vertex1, vertex2):
        edge1 = self.absolute_edge(edge, 0)
        edge2 = self.absolute_edge(edge, 1)

        if self.edges[edge1] == vertex1:
            if self.edges[edge2] == vertex2:
                return True

        if self.edges[edge1] == vertex2:
            if self.edges[edge2] == vertex1:
                return True

        return False

    def absolute_edge(self, edge, index):
        masked_edge = edge & self.edge_mask
        abs_edge = (masked_edge + (index * self.num_edges))
        return abs_edge

    def traverse_recursive(self, i):
        out = self.out
        self.visited.add(i)

        self.total_traversals += 1
        self.traversal_depth += 1
        depth = self.traversal_depth
        if depth > self.max_traversal_depth:
            self.max_traversal_depth = depth

        f = self.first[i]
        indent = '    ' * depth
        out(f'{indent}{i} -> {f}: begin traverse_recursive')

        iterator = GraphIterator(graph=self, vertex=i, depth=depth)

        count = 0
        index_mask = self.index_mask

        while True:

            count += 1
            neighbor = iterator.next_neighbor()

            if not neighbor:
                out(f'{indent}{i} -> {f}: [{count}/{depth}] no neighbor, break')
                break

            if neighbor in self.visited:
                out(f'{indent}{i} -> {f}: [{count}/{depth}] '
                    f'neighbor {neighbor} visited, continue')
                continue

            out(f'{indent}{i} -> {f}: [{count}/{depth}] '
                f'neighbor {neighbor} not visited, creating ID...')

            edge_id = self.edge_id(i, neighbor)

            masked_edge_id = edge_id & self.index_mask

            original_existing_id = existing_id = self.assigned[i]
            assert existing_id >= 0, existing_id

            this_id = edge_id - existing_id
            masked_this_id = this_id & index_mask
            assert masked_this_id <= self.num_vertices, masked_this_id

            final_id = edge_id + existing_id
            masked_final_id = final_id & index_mask
            assert masked_final_id <= self.num_vertices, masked_final_id

            bit = masked_final_id
            assert bit < self.num_vertices, (bit, num_vertices)

            if bit in self.index_bitmap:
                self.collisions += 1
            else:
                self.index_bitmap.add(bit)

            out(f'{indent}{i} -> {f}: [{count}/{depth}] '
                f'edge_id: {edge_id}, '
                f'masked_edge_id: {masked_edge_id}, '
                f'existing_id = {original_existing_id}, '
                f'this_id = {this_id}, '
                f'final_id = {final_id}, '
                f'masked_final_id/bit = {masked_final_id}')

            out(f'{indent}{i} -> {f}: [{count}/{depth}] '
                f'assigning neighbor {neighbor} id: {masked_this_id}')
            self.assigned[neighbor] = masked_this_id

            out(f'{indent}{i} -> {f}: [{count}/{depth}] '
                f'traversing neighbor {neighbor}')
            self.traverse_recursive(neighbor)

        self.traversal_depth -= 1

    def assign(self):
        assert self.hashed
        out = self.out
        for i in range(self.num_vertices):
            f = self.first[i]
            if is_empty(f):
                out(f'{i}: skipping, empty.')
                continue

            if f in self.visited:
                out(f'{i} -> {f}: skipping, already visited')
                continue

            out(f'{i} -> {f}: not visited, traversing...')
            self.traverse_recursive(i)


# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
