#%%
#CONVERT MODEL TO TFLITE
import tensorflow as tf

model = tf.keras.models.load_model('modelos/best_model.74-0.03.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
import pathlib

tflite_models_dir = pathlib.Path("models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"model.tflite"
tflite_model_file.write_bytes(tflite_model)


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




# %%
#convert using full integer quantization


def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((x_train)).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model_quant = converter.convert()
#%%
# Save the model.
tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)
# %%
# Comparamos modelos

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_indices):
  global x_test

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_indices),), dtype=int)
  for i, test_image_index in enumerate(test_indices):
    test_image = x_test[test_image_index]
    test_label = y_test[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

# %%

## Helper function to test the models on one image
def test_model(tflite_file, test_index,model_type):
  global y_test

  predictions = run_tflite_model(tflite_file, [test_index])
  #print(predictions)
  template = model_type + " Model \n True:" + str(y_test[test_index]) + ", Predicted:" + str(predictions)
  print(template)

# %%

test_index = 2000
#Modelos
test_model(tflite_model_file, test_index, model_type="Float")

test_model(tflite_model_quant_file, test_index, model_type="Quantized")

# %%
# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global x_test
  global y_test

  test_indices = range(x_test.shape[0])
  predictions = run_tflite_model(tflite_file, test_indices)

  accuracy = (np.sum(y_test== predictions) * 100) / len(x_test)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(x_test)))

# %%
evaluate_model(tflite_model_file, model_type="Float")

evaluate_model(tflite_model_quant_file, model_type="Quantized")
# %%
