
* create bcarray from numpy matrix, with chunklen for optimal speed

* rechunk existing bcarray for optimal dot product speed

* enable dot products between numpy matrices, producing bcarray result

* create bcarray result for 'bcarray . ndarray' input, by default

* get asv running



what happens when columns are bigger than thingy? thingy = (OPTIMAL_BSIZE = 2 ** 22) / (bytes = 2 ** 3 (for np.int64)) = 2 ** 19

what's the actual number for optimal size (more resolution than a power of two)?

how does optimal number compare to numpy?



## Concrete use cases

* Model prediction
	- large number of rows, small number of columns (~8 - 512 ( 2 ** 3 - 2 ** 9))

* Covariance matrix
	- https://en.wikipedia.org/wiki/Covariance_matrix
	- v . v.T (where v is N x 1)

N ~ 100,000 - 1,000,000,000 ( 2 ** 17 - 2 ** 30 )
M ~ 1 - 1024 ( 2 ** 0 - 2 ** 10)

Try to fit in 8GB

if you have enough rows and (optimal <= 2 ** 15) (sqrt of GB), do optimal
otherwise, do the biggest chunklen that is strictly less than pessimal

2 ** 15 is sqrt the of a Gigabyte, which is the length and width of the square matrix of 64 bit objects which takes up 8GB

come up with ten (10) explicit optimization tests (# cols, # rows, data type)
test those on the three (3) machines that you're using right now (and maybe Yan's)


Analytical ~1m
1. Outer Product on Vectors (Correlation Matrix)
	a. medium data		 51,810 rows
	b. pretty big data	366,357 rows

Production ~200ms
2. Nearest Neighbor Search
	a. medium length	 300,000 rows
		i.    32 columns
		ii.  128 columns
		iii. 512 columns
	b. big length		1,000,000 rows
		i.    32 columns
		ii.  128 columns
		iii. 512 columns
