# internal imports
from bdot import carray_ext

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def dot(self, matrix, output='ndarray', rootdir=None):

		# check dtype compatibility
		if self.dtype.type != matrix.dtype.type:
			raise ValueError("inputs must have the same dtype. Found {0} and {1}".format(self.dtype, matrix.dtype))

		# check shape
		if( (self.shape[1] != matrix.shape[0]) and 
			(type(matrix) == carray) and (self.shape[1] != matrix.shape[1])):
			raise ValueError("inputs must have compatible shapes. Found {0} and {1}".format(self.shape, matrix.shape))

		if type(matrix) == np.ndarray:
			if rootdir == None and output != 'carray':
				# output ndarray
				return carray_ext._dot(self, matrix)
			else:
				# output carray
				return carray_ext._dot_carray(self, matrix, rootdir)
		else:
				# output carray
			return carray_ext._dot_mat_carray(self, matrix, matrix[0], rootdir)
