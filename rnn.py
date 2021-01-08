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

def rnn_forward_pass(x, a0, parameters):
	caches = []
	
	_, m, T_x = x.shape
	n_y, n_a = parameters["Wya"].shape

	a = np.zeros((n_a, m, T_x))

	a_next = a0

	for t in range (T_x):
		a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
		a[:,:,t] = a_next
		y_pred[:,:,t] = a_next
	
	caches.append(cache)
	caches = (caches, x)

	return a, y_pred, caches
