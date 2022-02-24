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

# Tensorflow Quantum 
from cirq.contrib.svg import SVGCircuit
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
from keras.models import Sequential, load_model
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error

# classical preprocessing
# for this class we need only the path and filename for our classical data
# we consider two methos using only the closing values and normalzie or
# use PCA for delete osme parameter and use the original data
# we consider 30 (or indicate in the input) days per instance,
# is important that the size og the instance must be module 3
# And consider the method hold-out 70-30 to train,
# we can choose another values for this valid method.


class Normalization():
    def __init__(self, filename='AAPL', address='Data/stocks_predictions/', address_original='Data/Stocks/'): # read a default file
        self.filename = filename
        self.address = address
        self.address_original = address_original

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
        df = pd.read_csv(self.address_original+self.filename+'.csv') # using pandas to read the csv file
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
        
        
        df_output = pd.read_csv(self.address+self.filename+'_train.csv')
        predictions = scaler.inverse_transform(np.array(df_output['y_pred']).reshape(-1,1))
        y_test_scaled = scaler.inverse_transform(np.array(df_output['y_real']).reshape(-1, 1))

        results = np.dstack((y_test_scaled,predictions))
        print(results)
        return results
        

if __name__ == "__main__":
	name_stocks = ['AAPL','ABB','ABBV','TOT','WMT','DUK','CHL','HSBC']
	address='Data/stocks_predictions/'
	address_original='Data/Stocks/'
	for name in name_stocks:
		nm = Normalization(filename=name)
		np.save(address+'data_'+name+'.npy', nm)
