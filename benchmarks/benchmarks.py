# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.

    Production
    1. Nearest Neighbor Search
        a. medium length        300,000 rows
            i.   32 columns
            ii.  128 columns
            iii. 512 columns
        b. big length
            i.    32 columns
            ii.  128 columns
            iii. 512 columns

    Analytical
    2. Outer Product on Vectors (Correlation Matrix)
        a. medium data      51,810 rows
        b. pretty big data 366,357 rows
        
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_iterkeys(self):
        for key in self.d.iterkeys():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]

    def time_xrange(self):
        d = self.d
        for key in xrange(500):
            x = d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
