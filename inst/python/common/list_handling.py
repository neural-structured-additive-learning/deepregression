def lapply(seq, fun):

	return([fun(x) for x in seq])
	
def remove_nones(x):

	return([x for x in test if x is not None])
