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

	cdef Py_ssize_t i, chunklen, leftover_count

	leftover_count = cython.cdiv(matrix.leftover, matrix.atomsize)

	chunklen = matrix.chunklen

	for i in range(matrix.nchunks):
		chunk_ = matrix.chunks[i]

		chunk_._getitem(0, chunklen, m_i.data)
		dot_i = np.dot(m_i, vector)


		result[i*chunklen:(i+1)*chunklen] = dot_i

	if leftover_count > 0:
		dot_i = np.dot(matrix.leftover_array, vector)

		result[(i+1)*chunklen:] = dot_i[:leftover_count]


	return result