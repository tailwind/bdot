# internal imports
from bdot import carray_ext

# external imports
import numpy as np
import bcolz

class carray(bcolz.carray):

	def dot(self, vector):

		col_dtype = self.dtype

		return carray_ext._dot(self, vector)
