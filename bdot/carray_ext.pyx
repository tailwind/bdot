import numpy as np
cimport numpy as np

#from numpy cimport ndarray, dtype, npy_intp, npy_int32, \
#    npy_uint64, npy_int64, npy_float64, npy_bool

import cython
import bcolz as bz
from bcolz.carray_ext cimport carray, chunk


# numpy optimizations from:
# http://docs.cython.org/src/tutorial/numpy.html


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dot_float64(carray matrix, np.ndarray[np.float64_t, ndim=1] vector):


	return vector


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef dot_int64(carray matrix, np.ndarray[np.int64_t, ndim=1] vector):
	
	cdef np.ndarray[np.int64_t] dot_i = np.empty(matrix.chunklen, dtype=np.int64)

	cdef np.ndarray[np.int64_t, ndim=2] m_i = np.empty((matrix.chunklen, matrix.shape[1]), dtype=np.int64)

	cdef np.ndarray[np.int64_t] result = np.empty(matrix.shape[0], dtype=np.int64)

	cdef chunk chunk_

	cdef Py_ssize_t i, chunk_start, chunk_len, leftover_len
	cdef unsigned int result_j, j

	chunk_len = matrix.chunklen

	leftover_len = cython.cdiv(matrix.leftover, matrix.atomsize)


	for i in range(matrix.nchunks):
		chunk_ = matrix.chunks[i]

		chunk_._getitem(0, chunk_len, m_i.data)
		dot_i = np.dot(m_i, vector)

		# copy to result
		chunk_start = i * chunk_len
		for j in range(chunk_len):
			result_j = <unsigned int> (chunk_start + j)
			result[result_j] = dot_i[j]

	if leftover_len > 0:
		dot_i = np.dot(matrix.leftover_array, vector)

		chunk_start = (i + 1) * chunk_len
		for j in range(leftover_len):
			result_j = <unsigned int> (chunk_start + j)
			result[result_j] = dot_i[j]


	return result