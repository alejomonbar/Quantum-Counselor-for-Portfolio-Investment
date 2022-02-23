#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**


from tfqml import *


# ## **Choose the 8 stocks**

name_stocks = ['AAPL','ABB','ABBV','TOT','WMT','DUK','CHL','HSBC']


# # Classical Preprocessing
for name in name_stocks:
	cp = ClassicalPreprocessing(filename=name)
	days = 30
	x_train,y_train,x_test,y_test = cp.preprocessing(days=days,flag_pca=True)


	# # Quantum Pre processing


	qp = QuantumPreprocessing(x_train,x_test)
	x_trainq,x_testq = qp.data2qubits()


	# Print a quantum circuit of our encoding


	qp.print_circuit(0)


	qubits_required = days//3
	qm = QuantumModel(qubits_required)
	model_circuit, model_readout =qm.quantum_circuit([cirq.Y,cirq.X])
	SVGCircuit(model_circuit)


	# Creating our LSTM hibryd model:



	# Build the Keras model.
	model = tf.keras.Sequential([
	    # The input is the data-circuit, encoded as a tf.string
	    tf.keras.layers.Input(shape=(), dtype=tf.string),
	    # The PQC layer returns the expected value of the readout gate.
	    tfq.layers.PQC(model_circuit, model_readout),
	    tf.keras.layers.Reshape((1,10)),
	    tf.keras.layers.LSTM(units=100, return_sequences=True),
	    tf.keras.layers.Dropout(0.2),
	    tf.keras.layers.LSTM(units=100, return_sequences=True),
	    tf.keras.layers.Dropout(0.2),
	    tf.keras.layers.LSTM(units=5),
	    tf.keras.layers.Dense(units=1),
	])


	# In[8]:


	model.compile(
	    loss='mean_squared_error',
	    optimizer=tf.keras.optimizers.Adam(),   		     metrics=[tf.keras.metrics.RootMeanSquaredError()])





	qnn_history = model.fit(
	      x_trainq, y_train,
	      batch_size=32,
	      epochs=60)

	qnn_results = model.evaluate(x_testq, y_test)
	save_data(model,x_trainq,y_train,name+"_train")
	save_data(model,x_testq,y_test,name+"_test")





