import numpy as np

def rnn_cell_forward(xt, a_prev, parameters):
	Wax = parameters["Wax"]
	Waa = parameters["Waa"]
	Wya = parameters["Wya"]
	ba = parameters["ba"]
	by = parameters["by"]

	a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a+prev) + ba)

	yt_pred = softmax(np.dot(Wya, a_next) + by)

	cache = (a_next, a_prev, xt, parameters)

	return a_next, yt_pred, cache
