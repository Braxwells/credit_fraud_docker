import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow import keras as ks
from keras.layers import Dense

def train():

    #Importamos el archivo csv
    training = "./data/creditcard_2023.csv"
    data_training = pd.read_csv(training)

    #Dividimos nuestro csv en x e y para poder entrenar nuestro modelo
    x = data_training.iloc[:,1:-1].values
    y = data_training.iloc[:,-1].values

    #Separamos nuestros datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


    #Creamos la red
    model = ks.models.Sequential()

    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(32, activation= 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=60)

    #Guardamos el modelo
    from joblib import dump
    dump(model, './models/Inference_NN.joblib')


if __name__ == '__main__':
    train()