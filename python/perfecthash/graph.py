#===============================================================================
# Imports
#===============================================================================
import sys

import numpy as np
import pandas as pd

from collections import (
    defaultdict,
)

from perfecthash.util import (
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
    def __init__(self, keys, out=None, hash_all_func=None):
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
        self.order = np.zeros(num_keys, dtype=np.uint32)
        self.counts = np.zeros(num_vertices, dtype=np.uint32)
        self.assigned = np.zeros(num_vertices, dtype=np.uint32)
        #self.assigned = np.full(num_vertices, -1, dtype=np.int32)
        self.vertices1 = np.zeros(num_keys, dtype=np.uint32)
        self.vertices2 = np.zeros(num_keys, dtype=np.uint32)
        self.edge_mask = num_edges - 1
        self.index_mask = num_edges - 1
        self.hash_mask = num_vertices - 1
        self.vertex_mask = num_vertices - 1
        self.visited = set()
        self.deleted = set()
        self.collisions = 0
        self.order_index = -1
        self.index_bitmap = set()
        self.traversal_depth = 0
        self.total_traversals = 0
        self.max_traversal_depth = 0
        self.acyclic = None

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
            self.pairs,
            self.counts,
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

    def find_degree1_edge(self, vertex):
        edge = self.first[vertex]
        found = False
        return_edge = None

        if is_empty(edge):
            return False

        abs_edge = self.absolute_edge(edge, 0)

        assert abs_edge <= edge, (abs_edge, edge)

        if not self.is_deleted_edge(abs_edge):
            found = True
            return_edge = edge

        while True:
            edge = self.next[edge]
            if is_empty(edge):
                break

            abs_edge = self.absolute_edge(edge, 0)

            if self.is_deleted_edge(abs_edge):
                continue

            if found:
                return False

            return_edge = edge
            found = True

        return return_edge if found else False

    def is_deleted_edge(self, edge):
        return edge in self.deleted

    def register_edge_deletion(self, edge):
        assert edge not in self.deleted
        self.deleted.add(edge)
        self.order_index -= 1
        ix = self.order_index
        order = self.order
        assert ix >= 0, ix
        assert order[ix] == 0, order[ix]
        order[ix] = edge

    def cyclic_delete_edge(self, vertex):

        edge = self.find_degree1_edge(vertex)

        if edge is False:
            return

        vertex1 = vertex
        vertex2 = 0

        while True:

            abs_edge = self.absolute_edge(edge, 0)

            assert edge >= abs_edge, (edge, abs_edge)

            self.register_edge_deletion(abs_edge)

            vertex2 = self.edges[abs_edge]
            if vertex2 == vertex1:
                abs_edge = self.absolute_edge(edge, 1)
                vertex2 = self.edges[abs_edge]

            if is_empty(vertex2):
                break

            prev_edge = edge

            edge = self.find_degree1_edge(vertex2)
            if edge is False:
                break

            assert prev_edge != edge, (prev_edge, edge)
            vertex1 = vertex2

    def is_acyclic(self):
        if self.acyclic is not None:
            return self.acyclic

        self.order_index = self.num_keys

        for vertex in range(self.num_vertices):
            self.cyclic_delete_edge(vertex)

        self.acyclic = (self.order_index == 0)
        return self.acyclic

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

    def assign2(self):
        assert self.acyclic

        order = self.order
        edges = self.edges
        visited = self.visited
        assigned = self.assigned
        num_edges = self.num_edges
        index_mask = self.index_mask

        for i in range(self.num_keys):

            edge1 = order[i]
            edge2 = edge1 + num_edges

            vertex1 = edges[edge1]
            vertex2 = edges[edge2]

            if vertex1 not in visited:
                assigned2 = assigned[vertex2]
                assigned1 = (((num_edges + edge1) - assigned2) & index_mask)
                assigned[vertex1] = assigned1
            else:
                assigned1 = assigned[vertex1]
                assigned2 = (((num_edges + edge1) - assigned1) & index_mask)
                assigned[vertex2] = assigned2

            visited.add(vertex1)
            visited.add(vertex2)

    def assign3(self):
        index_mask = self.index_mask
        num_verts = self.num_vertices
        adjacent = defaultdict(list)
        assigned = num_verts * [0]
        visited = num_verts * [False]
        depths = {}
        iterations = 0

        for (edge_value, key) in enumerate(self.keys):
            (vertex1, vertex2) = self.hash_key(key)
            adjacent[vertex1].append((vertex2, edge_value))
            adjacent[vertex2].append((vertex1, edge_value))

        for root in range(num_verts):
            if is_empty(self.first[root]):
                continue

            if visited[root]:
                continue

            iterations += 1
            tovisit = [(None, root)]
            depth = 0
            while tovisit:
                (parent, vertex) = tovisit.pop()
                visited[vertex] = True

                skip = True
                for (neighbor, edge_value) in adjacent[vertex]:
                    if skip:
                        if neighbor == parent:
                            skip = False
                            continue

                    if visited[neighbor]:
                        return False

                    tovisit.append((vertex, neighbor))
                    depth += 1

                    ix = (edge_value - assigned[vertex]) & index_mask
                    assigned[neighbor] = ix

            depths[root] = depth

        for (edge_value, key) in enumerate(self.keys):
            (vertex1, vertex2) = self.hash_key(key)
            ix = (assigned[vertex1] + assigned[vertex2]) & index_mask
            assert edge_value == ix, (edge_value, ix)

        self.assigned3 = assigned
        self.adjacent = adjacent
        self.iterations = iterations
        return True

    def as_df(self):

        assigned1 = self.assigned[self.vertices1]
        assigned2 = self.assigned[self.vertices2]
        assigned12 = assigned1 + assigned2
        assigned = assigned12 & self.index_mask

        df = pd.DataFrame({
            'Key': self.keys,
            'Order': self.order,
            'Vertex1': self.vertices1,
            'Vertex1Count': self.counts[self.vertices1],
            'Assigned1': assigned1,
            'Vertex2': self.vertices2,
            'Vertex2Count': self.counts[self.vertices2],
            'Assigned2': assigned2,
            '(Assigned1+Assigned2)': assigned12,
            'Index': assigned,
        })

        return df

def test1(assign_version=2):
    keys_path = r'c:/src/perfecthash-keys/sys32/acpipmi-15.keys'
    fp = np.memmap(keys_path, dtype='uint32', mode='r')
    keys = fp
    num_keys = len(keys)
    seed1 = 2721747647
    seed2 = 630241707
    seed3_byte1 = 0x4
    seed3_byte2 = 0x11
    graph = Graph(keys, out=print)
    graph.hash(seed1, seed2, seed3_byte1, seed3_byte2)
    assert graph.is_acyclic()
    if assign_version == 1:
        graph.assign()
    elif assign_version == 2:
        graph.assign2()
    elif assign_version == 3:
        graph.assign3()
    else:
        raise ValueError(assign_version)
    return graph

if __name__ == '__main__':
    graph = test1(assign_version=3)

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
