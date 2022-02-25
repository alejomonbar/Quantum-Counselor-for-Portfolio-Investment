# data's libraries
# ==============================================================================
import numpy as np
import pandas as pd

# plots's libraries
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Classical preprocessing
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error


# Tensorflow Quantum 
from cirq.contrib.svg import SVGCircuit
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy

# classical preprocessing
# for this class we need only the path and filename for our classical data
# we consider two methos using only the closing values and normalzie or
# use PCA for delete osme parameter and use the original data
# we consider 30 (or indicate in the input) days per instance,
# is important that the size og the instance must be module 3
# And consider the method hold-out 70-30 to train,
# we can choose another values for this valid method.


class ClassicalPreprocessing():
    def __init__(self, filename='AAPL', address='Data/Stocks/'): # read a default file
        self.filename = filename
        self.address = address

    def create_dataset(self,df,days=30): # convert an array  to split in x and y sets
        x = []
        y = []
        for i in range(days, df.shape[0]):
            x.append(df[i-days:i,0])
            y.append(df[i,0])
        x = np.array(x)
        y = np.array(y)
        return x,y 
        
    def preprocessing(self,porcentage=0.7,days=30,flag_pca=False):
        ## format of CSV file: Date,Open,High,Low,Close,Adj Close,Volume
        df = pd.read_csv(self.address+self.filename+'.csv') # using pandas to read the csv file
        del df["Date"] #delete the column Date
        df_preprocessing = []
        if flag_pca:
            df_pca = df.copy()
            for i in df_pca.columns: #apply pca methods
                df_pca[i] = MinMaxScaler().fit_transform(np.array(df_pca[i]).reshape(-1,1))
            pca_pipe = make_pipeline(StandardScaler(), PCA())
            pca_pipe.fit(df_pca)

            # Se extrae el modelo entrenado del pipeline
            model_pca = pca_pipe.named_steps['pca']
            
            pd.DataFrame(
            data    = model_pca.components_,
            columns = df_pca.columns,
            index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
            )
            df_pca = df_pca['Open'].values
            df_preprocessing = df_pca.reshape(-1, 1)
        else:
            df = df['Close'].values
            df_preprocessing = df.reshape(-1, 1)
            
        dataset_train = np.array(df_preprocessing[:int(df_preprocessing.shape[0]*porcentage)])
        dataset_test = np.array(df_preprocessing[int(df_preprocessing.shape[0]*porcentage):])
        
        scaler = MinMaxScaler(feature_range=(0,1))
        dataset_train = scaler.fit_transform(dataset_train)
        dataset_test = scaler.transform(dataset_test)

        #shows the shape of each train and data set
        print(f"Shape of train data: {dataset_train.shape}")
        print(f"Shape of test data: {dataset_test.shape}")
        
        x_train,y_train = self.create_dataset(dataset_train,days=days)   
        x_test,y_test = self.create_dataset(dataset_test,days=days)   
        
        # finish the classical preprocessing to obtain the x and y sets
        return x_train,y_train,x_test,y_test
        
        
#For this section we use Tensorflow Quantum to generate a quantum circuit to encode 
# the instances of our set in terms of a quantum circuit, the encoding is done by angle
# and modules of 3 are considered for each qubit.


class QuantumPreprocessing(): #consider onyl the x_train and x_test
    def __init__(self, x_train,x_test):
        self.x_train = x_train
        self.x_test = x_test
        
    def convert2circuit(self,values):
        #Encode classical data into quantum circuit
        num_qubits = len(values)//3
        qubits = cirq.GridQubit.rect(num_qubits, 1)
        circuit = cirq.Circuit()
        
        #we use an angle encoding use as reference https://arxiv.org/pdf/2009.01783.pdf  for the encoding 
        for i in range(num_qubits): 
            circuit.append(cirq.Y(qubits[i])**(np.arctan(values[i])))
            circuit.append(cirq.H(qubits[i]))
            circuit.append(cirq.X(qubits[i])**(np.arctan(values[i+num_qubits])))

        for i in  range((num_qubits//2)):
            circuit.append(cirq.CX(qubits[2*i],qubits[2*i+1]))

        for i in range(num_qubits):
            circuit.append(cirq.Z(qubits[i])**(np.arctan(values[2*num_qubits+i]))) 
        return circuit
    
    
    # covnert the quantum circuit into a tensor for use the classical methods for tf
    def data2qubits(self):
        x_train_circ = [self.convert2circuit(x) for x in self.x_train]
        x_test_circ = [self.convert2circuit(x) for x in self.x_test]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        return x_train_tfcirc, x_test_tfcirc
    
    # print the quantum circuit for the num instance
    def print_circuit(self,num):
        return SVGCircuit(self.convert2circuit(self.x_train[num]))
        
        
        
# class to generate a variational quantum circuit following our model, 
#for this it must compact with the same number of inputs
#of the classical encoding processing.

class CircuitLayer():
    def __init__(self, quantum_bits):
        self.quantum_bits = quantum_bits

    # generate a layer of our ansatz/ or variational quantum circuit design
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.quantum_bits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit)**symbol)

        num_qubits = len(self.quantum_bits)
        for i in  range((num_qubits//2)):
            circuit.append(cirq.CX(self.quantum_bits[2*i],self.quantum_bits[2*i+1]))

        
        for i, qubit in enumerate(self.quantum_bits):
            symbol = sympy.Symbol(prefix + '-' + str(i+num_qubits))
            if i%2 == 1:
                circuit.append(gate(qubit)**symbol)
                
        for i, qubit in enumerate(self.quantum_bits):
            symbol = sympy.Symbol(prefix + '-' + str(i+num_qubits+num_qubits//2))
            if i%2 == 1:
                circuit.append(cirq.H(self.quantum_bits[i]))           
                
                
# quantum circuit to generate our quantum neural network or variational quantum circuit,
# this follows a general scheme, but a layer of gates is considered changing, 
# these can be the X,Y,Z gates with some rotation with respect to the angle
# and the same number of qubits are measured to a vector
# of the same size with respect to the Z axis.

class QuantumModel():
    def __init__(self, qubits_required):
        self.qubits_required = qubits_required
        
    def quantum_circuit(self,pauli_list):
        #Create a QNN model
        quantum_bits = cirq.GridQubit.rect(self.qubits_required, 1)  
        classical_bits = cirq.GridQubit.rect(self.qubits_required,1 )      
        circuit = cirq.Circuit()

        builder = CircuitLayer(quantum_bits)

        # Then add n layers consider a input a gate, that could be cirq.X,cirq.Y,cirq.Z
        for pl in range(len(pauli_list)):
            builder.add_layer(circuit, pauli_list[pl], str(pl))

        # Finally, prepare the classical qubit with respect Z.
        list_classical_bits = []
        for i in range(self.qubits_required):
            list_classical_bits.append(cirq.Z(classical_bits[i]))

        return circuit,list_classical_bits
        
        
        
# print in a plot the result our  proposal hibryd model
def visualization(model,x_data_set,y_data_set):
    predictions = model.predict(x_data_set)
    predictions = predictions.reshape(len(y_data_set),1)

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#001340')
    ax.plot(y_data_set, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()

# save the data with the real and predict values

def save_data(model,x_data_set,y_data_set,filename):
    size = len(y_data_set)
    predictions = model.predict(x_data_set)
    predictions = predictions.reshape(size,1)
    
    mae = "{:.2f}".format(mean_absolute_error(y_data_set, predictions)*100)
    print(f"MAE {mae}%")
    mae_list = [mae]* size
    df = pd.DataFrame({'y_real':y_data_set.reshape(size), 'y_pred':predictions.reshape(size), 'MAE':mae_list})
    df.to_csv(filename+'.csv', index=False)
