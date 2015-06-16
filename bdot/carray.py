# internal imports
from bdot import carray_ext

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def dot(self, vector):

		# check dtype compatibility
		if self.dtype.type != vector.dtype.type:
			raise ValueError("inputs must have the same dtype. Found {0} and {1}".format(self.dtype, vector.dtype))

		# check shape

		return carray_ext._dot(self, vector)
