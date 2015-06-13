# internal imports
from bdot import carray_ext

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def dot(self, vector):

		col_dtype = self.dtype

		if col_dtype == np.float64:
			return carray_ext.dot_float64(self, vector)

		elif col_dtype == np.int64:
			return carray_ext.dot_int64(self, vector)

