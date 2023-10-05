
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import pandas as pd

from joblib import load
from sklearn.model_selection import train_test_split



def cambio_etiqueta(etiqueta_binaria):
    if etiqueta_binaria == 0:
        return "Verdadero"
    else:
        return "Fraudulento"

def inference():V
    # Importamos el archivo csv
    training = "./data/creditcard_2023.csv"
    data_training = pd.read_csv(training)

    # Dividimos nuestro csv en x e y para poder entrenar nuestro modelo
    x = data_training.iloc[:, 1:-1].values
    y = data_training.iloc[:, -1].values

    # Separamos nuestros datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    # Runeamos el modelo
    model = load('./models/Inference_NN.joblib')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Precisión del modelo en los datos de prueba:', test_acc)

    # Predecimos
    prob = model.predict(x_test)
    umbral = 0.99
    predicciones = (prob > umbral).astype(int)
    predicciones_clase = [cambio_etiqueta(etiqueta_binaria) for etiqueta_binaria in predicciones]
    print(predicciones_clase)

if __name__ == '__main__':
    inference()

