import numpy as np
cimport numpy as np
ctypedef unsigned int ULONG
ctypedef unsigned int *PULONG
ctypedef unsigned long long ULONGLONG
ctypedef ULONGLONG *PULONGLONG
ctypedef unsigned char BYTE
cdef struct VERTEX_PAIR:
    ULONG Vertex1
    ULONG Vertex2
ctypedef VERTEX_PAIR *PVERTEX_PAIR
from cython cimport boundscheck, wraparound
@boundscheck(False)
@wraparound(False)
cpdef hash_all(
    ULONG num_keys,
    ULONG num_edges,
    ULONG hash_mask,
    ULONG seed1,
    ULONG seed2,
    BYTE seed3_byte1,
    BYTE seed3_byte2,
    np.ndarray[np.uint32_t, ndim=1] keys,
    np.ndarray[np.uint32_t, ndim=1] vertices1,
    np.ndarray[np.uint32_t, ndim=1] vertices2,
    np.ndarray[np.int32_t, ndim=1] first,
    np.ndarray[np.int32_t, ndim=1] next_,
    np.ndarray[np.int32_t, ndim=1] edges,
    np.ndarray[np.uint32_t, ndim=1] pairs,
    np.ndarray[np.uint32_t, ndim=1] counts,
):
    cdef ULONG i = 0
    cdef ULONG k = 0
    cdef ULONG key
    cdef ULONG edge1
    cdef ULONG edge2
    cdef ULONG first1
    cdef ULONG first2
    cdef ULONG vertex1
    cdef ULONG vertex2
    for i in range(num_keys):
        key = keys[i]
        vertex1 = (((key * seed1) >> seed3_byte1) & hash_mask)
        vertex2 = (((key * seed2) >> seed3_byte2) & hash_mask)
        vertices1[i] = vertex1
        vertices2[i] = vertex2
        pairs[k] = vertex1
        k += 1
        pairs[k] = vertex2
        k += 1

        counts[vertex1] += 1
        counts[vertex2] += 1

        edge1 = i
        edge2 = i + num_edges

        first1 = first[vertex1]
        next_[edge1] = first1
        first[vertex1] = edge1
        edges[edge1] = vertex2

        first2 = first[vertex2]
        next_[edge2] = first2
        first[vertex2] = edge2
        edges[edge2] = vertex1

cpdef hash_v1(ULONG key,
              ULONG hash_mask,
              ULONG seed1,
              ULONG seed2,
              ULONG seed3_byte1,
              ULONG seed3_byte2):
    cdef ULONG vertex1
    cdef ULONG vertex2
    vertex1 = (((key * seed1) >> seed3_byte1) & hash_mask)
    vertex2 = (((key * seed2) >> seed3_byte2) & hash_mask)
    return vertex1

cpdef hash_key(
    ULONG key,
    ULONG hash_mask,
    ULONG seed1,
    ULONG seed2,
    ULONG seed3_byte1,
    ULONG seed3_byte2
):
    cdef ULONG vertex1
    cdef ULONG vertex2
    vertex1 = (((key * seed1) >> seed3_byte1) & hash_mask)
    vertex2 = (((key * seed2) >> seed3_byte2) & hash_mask)
    return (vertex1, vertex2)


cpdef hash_v1_parts(ULONG key,
              ULONG hash_mask,
              ULONG seed1,
              ULONG seed2,
              ULONG seed3_byte1,
              ULONG seed3_byte2):
    cdef ULONG vertex1
    cdef ULONG vertex2
    cdef ULONG part1
    cdef ULONG part2
    cdef ULONG part3
    part1 = key * seed1
    part2 = part1 >> seed3_byte1
    part3 = part2 & hash_mask
    return (part1, part2, part3)

cpdef hash_v2(ULONG key,
              ULONG hash_mask,
              ULONG seed1,
              ULONG seed2,
              ULONG seed3_byte1,
              ULONG seed3_byte2):
    cdef ULONG vertex1
    cdef ULONG vertex2
    vertex1 = (((key * seed1) >> seed3_byte1) & hash_mask)
    vertex2 = (((key * seed2) >> seed3_byte2) & hash_mask)
    return vertex2
