# internal imports
from bdot import carray_ext

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def dot(self, matrix):

		# check dtype compatibility
		if self.dtype.type != matrix.dtype.type:
			raise ValueError("inputs must have the same dtype. Found {0} and {1}".format(self.dtype, matrix.dtype))

		# check shape
		if not self.shape[1] == matrix.shape[0]:
			raise ValueError("inputs must have compatible shapes. Found {0} and {1}".format(self.shape, matrix.shape))

		return carray_ext._dot(self, matrix)
