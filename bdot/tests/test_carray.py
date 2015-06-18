import nose
from nose.tools import raises

import bdot
import bcolz
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

def test_dot_int64():

	matrix = np.random.random_integers(0, 12000, size=(10000, 100))
	bcarray = bdot.carray(matrix, chunklen=2**13, cparams=bcolz.cparams(clevel=2))

	v = bcarray[0]

	result = bcarray.dot(v)
	expected = matrix.dot(v)

	assert_array_equal(expected, result)


def test_dot_int32():

	matrix = np.random.random_integers(0, 12000, size=(10000, 100)).astype('int32')
	bcarray = bdot.carray(matrix, chunklen=2**13, cparams=bcolz.cparams(clevel=2))

	v = bcarray[0]

	result = bcarray.dot(v)
	expected = matrix.dot(v)

	assert_array_equal(expected, result)


def test_dot_float64():

	matrix = np.random.random_sample(size=(10000, 100))
	bcarray = bdot.carray(matrix, chunklen=2**13, cparams=bcolz.cparams(clevel=2))

	v = bcarray[0]

	result = bcarray.dot(v)
	expected = matrix.dot(v)

	assert_array_almost_equal(expected, result, decimal=5)


def test_dot_float32():

	matrix = np.random.random_sample(size=(10000, 100)).astype('float32')
	bcarray = bdot.carray(matrix, chunklen=2**13, cparams=bcolz.cparams(clevel=2))

	v = bcarray[0]

	result = bcarray.dot(v)
	expected = matrix.dot(v)

	assert_array_almost_equal(expected, result, decimal=5)


def test_dot_matrix_int64():

	matrix = np.random.random_integers(0, 120, size=(1000, 100))
	bcarray1 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	bcarray2 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix.dot(matrix.T)

	assert_array_equal(expected, result)


def test_dot_matrix_int64_unequal_chunklen():

	matrix1 = np.random.random_integers(0, 120, size=(1000, 100))
	bcarray1 = bdot.carray(matrix1, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	matrix2 = np.random.random_integers(0, 120, size=(1000, 100))
	bcarray2 = bdot.carray(matrix2, chunklen=2**8, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix1.dot(matrix2.T)

	assert_array_equal(expected, result)


def test_dot_matrix_int64_unequal_length():

	matrix1 = np.random.random_integers(0, 120, size=(1000, 100))
	bcarray1 = bdot.carray(matrix1, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	matrix2 = np.random.random_integers(0, 120, size=(10000, 100))
	bcarray2 = bdot.carray(matrix2, chunklen=2**10, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix1.dot(matrix2.T)

	assert_array_equal(expected, result)

def test_dot_matrix_int32():

	matrix = np.random.random_integers(0, 120, size=(1000, 100)).astype('int32')
	bcarray1 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	bcarray2 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix.dot(matrix.T)

	assert_array_equal(expected, result)


def test_dot_matrix_float64():

	matrix = np.random.random_sample(size=(1000, 100))
	bcarray1 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	bcarray2 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix.dot(matrix.T)

	assert_array_almost_equal(expected, result, decimal=5)


def test_dot_matrix_float32():

	matrix = np.random.random_sample(size=(1000, 100)).astype('float32')
	bcarray1 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))
	bcarray2 = bdot.carray(matrix, chunklen=2**9, cparams=bcolz.cparams(clevel=2))


	result = bcarray1.dot(bcarray2)
	expected = matrix.dot(matrix.T)

	assert_array_almost_equal(expected, result, decimal=4)


@raises(ValueError)
def test_dot_incompatible_dtype():

	matrix = np.random.random_integers(0, 12000, size=(10000, 100))
	bcarray = bdot.carray(matrix, chunklen=2**13, cparams=bcolz.cparams(clevel=2))

	v = bcarray[0].astype('int32')

	result = bcarray.dot(v)

'''
@raises(ValueError)
def test_dot_incompatible_shapes():

	matrix = np.random.random_integers(0, 12000, size=(10000, 101))
	bcarray = bdot.carray(matrix[:, :100], chunklen=2**13, cparams=bcolz.cparams(clevel=2))


	result = bcarray.dot(matrix)
'''
