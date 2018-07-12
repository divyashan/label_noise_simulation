import pdb
import numpy as np
import pandas as pd

from simulation import run_test

n_example_opts = [1000]
delta_0_opts = [0]
delta_1_opts = [.2]
p_opts = [.1, .25, .5]


for n_examples in n_example_opts:
	for delta_0 in delta_0_opts:
		for delta_1 in delta_1_opts:
			for p in p_opts:
				run_test(p, delta_0, delta_1, n_examples)
	print "N Examples: ", n_examples
