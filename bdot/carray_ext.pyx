import numpy as np
cimport numpy as np

import cython
cimport cython
import bcolz as bz
from bcolz.carray_ext cimport carray, chunk


# numpy optimizations from:
# http://docs.cython.org/src/tutorial/numpy.html

# fused types (templating) from
# http://docs.cython.org/src/userguide/fusedtypes.html

ctypedef fused numpy_native_number:
	np.int64_t
	np.int32_t
	np.float64_t
	np.float32_t


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _dot(carray matrix, np.ndarray[numpy_native_number, ndim=1] vector):
	'''
		Calculate dot product between a bcolz.carray matrix and a numpy vector.
		Second dimension of matrix must match first dimension of vector.
		
		Arguments:
			matrix (carray): two dimensional matrix in a bcolz.carray, row vector format
			vector (ndarray): one dimensional vector in a numpy array
		
		Returns:
			ndarray: result of dot product, one value per row in orginal matrix
	'''

	# fused type conversion
	if numpy_native_number is np.int64_t:
		p_dtype = np.int64
	elif numpy_native_number is np.int32_t:
		p_dtype = np.int32
	elif numpy_native_number is np.float64_t:
		p_dtype = np.float64
	else:
		p_dtype = np.float32

	cdef np.ndarray[numpy_native_number] dot_i = np.empty(matrix.chunklen, dtype=p_dtype)

	cdef np.ndarray[numpy_native_number, ndim=2] m_i = np.empty((matrix.chunklen, matrix.shape[1]), dtype=p_dtype)

	cdef np.ndarray[numpy_native_number] result = np.empty(matrix.shape[0], dtype=p_dtype)

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


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _dot_mat(carray m1, carray m2, np.ndarray[numpy_native_number, ndim=1] type_indicator):
	'''
		Calculate matrix multiply between bcolz.carray matrix and transpose of
		a second bcolz.carray matrix.
		Second dimension of m1 must match second dimension of m2.
		
		Requires that resulting matrix fit in RAM.

		Requires that chunks and matrix multiply of chunks fit in RAM.

		Arguments:
			m1 (carray): two dimensional matrix in a bcolz.carray, row vector format
			m2 (carray): two dimensional matrix in a bcolz.carray, row vector format
			type_indicator(ndarray) : hack to allow use of fused types (just pass in the first row)
		
		Returns:
			carray: result of matrix multiply between m1 and m2.T, matrix with dimensions equal to 
			first dimension of m1 by first dimension of m2

	'''

	# fused type conversion
	if numpy_native_number is np.int64_t:
		p_dtype = np.int64
	elif numpy_native_number is np.int32_t:
		p_dtype = np.int32
	elif numpy_native_number is np.float64_t:
		p_dtype = np.float64
	else:
		p_dtype = np.float32

	cdef Py_ssize_t i, chunk_start_i, chunk_len_i, leftover_len_i
	cdef Py_ssize_t j, chunk_start_j, chunk_len_j, leftover_len_j

	# iterate through m2 in outer loop, to facilitate later chunking into output carray
	chunk_len_i = m2.chunklen
	chunk_len_j = m1.chunklen


	cdef np.ndarray[numpy_native_number, ndim=2] m_i = np.empty((chunk_len_i, m2.shape[1]), dtype=p_dtype)

	cdef np.ndarray[numpy_native_number, ndim=2] m_j = np.empty((chunk_len_j, m1.shape[1]), dtype=p_dtype)

	cdef np.ndarray[numpy_native_number, ndim=2] dot_k = np.empty((chunk_len_i, chunk_len_j), dtype=p_dtype)

	cdef np.ndarray[numpy_native_number, ndim=2] result = np.empty((m1.shape[0], m2.shape[0]), dtype=p_dtype)

	cdef chunk chunk_i_
	cdef chunk chunk_j_

	cdef unsigned int result_k, k
	cdef unsigned int result_l, l


	leftover_len_i = cython.cdiv(m2.leftover, m2.atomsize)
	leftover_len_j = cython.cdiv(m1.leftover, m1.atomsize)


	for i in range(m2.nchunks):

		chunk_i_ = m2.chunks[i]
		chunk_i_._getitem(0, chunk_len_i, m_i.data)


		for j in range(m1.nchunks):
			chunk_j_ = m1.chunks[j]

			chunk_j_._getitem(0, chunk_len_j, m_j.data)

			dot_k = np.dot(m_j, m_i.T)

			# copy to result
			chunk_start_i = i * chunk_len_i
			chunk_start_j = j * chunk_len_j
			for k in range(chunk_len_i):
				result_k = <unsigned int> (chunk_start_i + k)
				for l in range(chunk_len_j):
					result_l = <unsigned int> (chunk_start_j + l)
					result[result_l, result_k] = dot_k[l, k]

		# do last chunk in first array
		if leftover_len_j > 0:
			dot_k = np.dot(m1.leftover_array, m_i.T)

			chunk_start_i = i * chunk_len_i
			chunk_start_j = (j + 1) * chunk_len_j
			for k in range(chunk_len_i):
				result_k = <unsigned int> (chunk_start_i + k)
				for l in range(leftover_len_j):
					result_l = <unsigned int> (chunk_start_j + l)
					result[result_l, result_k] = dot_k[l, k]


	# do last chunk in second array
	if leftover_len_i > 0:

		for j in range(m1.nchunks):

			chunk_j_ = m1.chunks[j]

			chunk_j_._getitem(0, chunk_len_j, m_j.data)

			dot_k = np.dot(m_j, m2.leftover_array.T)

			# copy to result
			chunk_start_i = (i + 1) * chunk_len_i
			chunk_start_j = j * chunk_len_j
			for k in range(leftover_len_i):
				result_k = <unsigned int> (chunk_start_i + k)
				for l in range(chunk_len_j):
					result_l = <unsigned int> (chunk_start_j + l)
					result[result_l, result_k] = dot_k[l, k]


		# do last chunk in first array
		if leftover_len_j > 0:
			dot_k = np.dot(m1.leftover_array, m2.leftover_array.T)

			chunk_start_i = (i + 1) * chunk_len_i
			chunk_start_j = (j + 1) * chunk_len_j
			for k in range(leftover_len_i):
				result_k = <unsigned int> (chunk_start_i + k)
				for l in range(leftover_len_j):
					result_l = <unsigned int> (chunk_start_j + l)
					result[result_l, result_k] = dot_k[l, k]


	return result
