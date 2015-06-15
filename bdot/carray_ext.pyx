import numpy as np
cimport numpy as np

import cython
import bcolz as bz
from bcolz.carray_ext cimport carray, chunk


# numpy optimizations from:
# http://docs.cython.org/src/tutorial/numpy.html

# fused types (templating) from
# http://docs.cython.org/src/userguide/fusedtypes.html

ctypedef fused int_or_float:
	np.int64_t
	np.int32_t
	np.float64_t
	np.float32_t


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _dot(carray matrix, np.ndarray[int_or_float, ndim=1] vector):

	if int_or_float is np.int64_t:
		p_dtype = np.int64
	elif int_or_float is np.int32_t:
		p_dtype = np.int32
	elif int_or_float is np.float64_t:
		p_dtype = np.float64
	else:
		p_dtype = np.float32

	cdef np.ndarray[int_or_float] dot_i = np.empty(matrix.chunklen, dtype=p_dtype)

	cdef np.ndarray[int_or_float, ndim=2] m_i = np.empty((matrix.chunklen, matrix.shape[1]), dtype=p_dtype)

	cdef np.ndarray[int_or_float] result = np.empty(matrix.shape[0], dtype=p_dtype)

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