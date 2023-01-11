#%%
#READ DATA FROM ONE TYPE OF SIGNAL
import pandas as pd
import glob
import os

COLUMNS = ['t_sample', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
#TODO si cojemos 2 segundos de movimiento el numero de muestras es 28 por gesto
NGESTOS = 28
STEP_DISTANCE = 14 #cada cuantas muestras pasamos la secuencia si de 28 en 28 o de 14 en 14...

#read all files with the name anular in the folder datos and read them and concat in one df
path = r'datos' # use your path
all_files = glob.glob(os.path.join(path, "anular*.csv"))

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df = df.T
    li.append(df)

anular = pd.concat(li, axis=0, ignore_index=True)
anular.columns = COLUMNS
#drop t_sample column
anular = anular.drop(columns=['t_sample'])
#ensure that we have a number of rows multiple of NGESTOS
anular = anular.iloc[:anular.shape[0] - anular.shape[0] % NGESTOS, :]
#add a column with the name of the class
anular['ref_signal'] = 'anular'
#separate 80% of the data for training and 20% for testing
anular_train = anular.iloc[:int(anular.shape[0]*0.8),:]
anular_test = anular.iloc[int(anular.shape[0]*0.8):,:]


# %%
#DO THE SAME WITH THE OTHER CLASSES

#read all files with the name pasos in the folder datos and read them and concat in one df
path = r'datos' # use your path
all_files = glob.glob(os.path.join(path, "pasos*.csv"))

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df = df.T
    li.append(df)

pasos = pd.concat(li, axis=0, ignore_index=True)
pasos.columns = COLUMNS
pasos = pasos.drop(columns=['t_sample'])
pasos = pasos.iloc[:pasos.shape[0] - pasos.shape[0] % NGESTOS, :]
pasos['ref_signal'] = 'pasos'
pasos_train = pasos.iloc[:int(pasos.shape[0]*0.8),:]
pasos_test = pasos.iloc[int(pasos.shape[0]*0.8):,:]

#tecnica
all_files = glob.glob(os.path.join(path, "tecnica*.csv"))

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df = df.T
    li.append(df)

tecnica = pd.concat(li, axis=0, ignore_index=True)
tecnica.columns = COLUMNS
tecnica = tecnica.drop(columns=['t_sample'])
tecnica = tecnica.iloc[:tecnica.shape[0] - tecnica.shape[0] % NGESTOS, :]
tecnica['ref_signal'] = 'tecnica'
tecnica_train = tecnica.iloc[:int(tecnica.shape[0]*0.8),:]
tecnica_test = tecnica.iloc[int(tecnica.shape[0]*0.8):,:]

#ataque
all_files = glob.glob(os.path.join(path, "ataque*.csv"))

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df = df.T
    li.append(df)

ataque = pd.concat(li, axis=0, ignore_index=True)
ataque.columns = COLUMNS
ataque = ataque.drop(columns=['t_sample'])
ataque = ataque.iloc[:ataque.shape[0] - ataque.shape[0] % NGESTOS, :]
ataque['ref_signal'] = 'ataque'
ataque_train = ataque.iloc[:int(ataque.shape[0]*0.8),:]
ataque_test = ataque.iloc[int(ataque.shape[0]*0.8):,:]


# %%
#CONCAT ALL DATA IN ONE DATAFRAME FOR TRAIN AND OTHER FOR TEST
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.concat([tecnica_train, pasos_train, ataque_train, anular_train], axis=0, ignore_index=True)
df_test = pd.concat([ tecnica_test, pasos_test, ataque_test, anular_test], axis=0, ignore_index=True)

ref_signals = df_train['ref_signal'].value_counts()
plt.bar(range(len(ref_signals)), ref_signals.values)
plt.xticks(range(len(ref_signals)), ref_signals.index)

#%%
#VISUALIZAMOS LOS DATOS

def dibuja_datos_aceleracion(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["accel_x"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["accel_y"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["accel_z"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

def dibuja_datos_giroscopio(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["gyro_x"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["gyro_y"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["gyro_z"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Gyro Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

for ref_signal in np.unique(df_train['ref_signal']):
    subset = df_train[df_train['ref_signal'] == ref_signal][:NGESTOS]
    dibuja_datos_aceleracion(subset, ref_signal)
    dibuja_datos_giroscopio(subset, ref_signal)


# %%
#NORMALIZAMOS LOS DATOS (no le gusta)
#normalize the data less the time and the reference signal columns
# df_train["accel_x"] = (df_train["accel_x"] - min(df_train["accel_x"].values)) / (max(df_train["accel_x"].values) - min(df_train["accel_x"].values))
# df_train["accel_y"] = (df_train["accel_y"] - min(df_train["accel_y"].values)) / (max(df_train["accel_y"].values) - min(df_train["accel_y"].values))
# df_train["accel_z"] = (df_train["accel_z"] - min(df_train["accel_z"].values)) / (max(df_train["accel_z"].values) - min(df_train["accel_z"].values))

# df_train["gyro_x"] = (df_train["gyro_x"] - min(df_train["gyro_x"].values)) / (max(df_train["gyro_x"].values) - min(df_train["gyro_x"].values))
# df_train["gyro_y"] = (df_train["gyro_y"] - min(df_train["gyro_y"].values)) / (max(df_train["gyro_y"].values) - min(df_train["gyro_y"].values))
# df_train["gyro_z"] = (df_train["gyro_z"] - min(df_train["gyro_z"].values)) / (max(df_train["gyro_z"].values) - min(df_train["gyro_z"].values))

# df_test["accel_x"] = (df_test["accel_x"] - min(df_test["accel_x"].values)) / (max(df_test["accel_x"].values) - min(df_test["accel_x"].values))
# df_test["accel_y"] = (df_test["accel_y"] - min(df_test["accel_y"].values)) / (max(df_test["accel_y"].values) - min(df_test["accel_y"].values))
# df_test["accel_z"] = (df_test["accel_z"] - min(df_test["accel_z"].values)) / (max(df_test["accel_z"].values) - min(df_test["accel_z"].values))

# df_test["gyro_x"] = (df_test["gyro_x"] - min(df_test["gyro_x"].values)) / (max(df_test["gyro_x"].values) - min(df_test["gyro_x"].values))
# df_test["gyro_y"] = (df_test["gyro_y"] - min(df_test["gyro_y"].values)) / (max(df_test["gyro_y"].values) - min(df_test["gyro_y"].values))
# df_test["gyro_z"] = (df_test["gyro_z"] - min(df_test["gyro_z"].values)) / (max(df_test["gyro_z"].values) - min(df_test["gyro_z"].values))


#comprobar que se ha hecho bien
plt.figure(figsize=(5,5))
plt.plot(df_train["accel_x"].values[:28])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")

# %%
#HACEMOS EL LABEL ENCODING
from sklearn import preprocessing

LABEL = 'RefSignalEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df_train[LABEL] = le.fit_transform(df_train['ref_signal'].values.ravel())
df_test[LABEL] = le.fit_transform(df_test['ref_signal'].values.ravel())

print(df_train.head())
print(df_test.head())

# %%
#CREAMOS LAS SECUENCIAS DE 28 GESTOS
#%% Creamos las secuencias

from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z accel and gyro features
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        accel_x = df['accel_x'].values[i: i + time_steps]
        accel_y = df['accel_y'].values[i: i + time_steps]
        accel_z = df['accel_z'].values[i: i + time_steps]
        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]
        # Lo etiquetamos como la actividad más frecuente 
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              NGESTOS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              NGESTOS,
                                              STEP_DISTANCE,
                                              LABEL)



#%%
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np

# modelo, etiqeutas e imagen de ejemplo
model_file = "prueba_Antonio.tflite"
labels = ['anular','ataque','pasos','tecnica']

#%%

# Inicializar el interprete de tensorflow lite
interpreter = edgetpu.make_interpreter(model_file)
# reserva de los tensores
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
# Ajustamos datos entrada
# size = common.input_size(interpreter) # datos entrada

# Cargamos la muestra que vamos a usar
prueba = x_test[1405].reshape(1, 28, 6)
# Ejecutar inferencia
common.set_input(interpreter, prueba)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=4)

# Comprobamos el resultados
repuesta = classes[0].id
print(labels[repuesta])
# %%
