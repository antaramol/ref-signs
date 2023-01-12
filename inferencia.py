import numpy as np
import pandas as pd
import time
import statistics as stats
import os.path
from time import sleep
from sense_hat import SenseHat
sense=SenseHat()

t=[]
t_sample=[]
accel_x, accel_y, accel_z = [], [], []
gyros_x, gyros_y, gyros_z = [], [], []
t_ini=time.time()
t_ant=t_ini

muestras=28 # muestras que tomamos, 1 min son 1600
accel_average=[]

import tflite_runtime.interpreter as tflite


# Initialize the interpreter
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("No es el modelo cuantizado:")
print('input: ', input_details['dtype'])
print('output: ', output_details['dtype'])


labels = ['anular','ataque','pasos','tecnica']
prediction = np.zeros(len(labels),dtype=int)


archivo="ataque" # nombre del archivo
tipo_archivo=".csv" # extensión 

print("grabando")
#bucle for para tomar los datos
while(1):
    for i in range(muestras):
        #print(i)
        #t_actual=time.time()
        acceleration = sense.get_accelerometer_raw()
        #datos de aceleración en Gs
        accel_x.append(acceleration['x'])
        accel_y.append(acceleration['y'])
        accel_z.append(acceleration['z'])
        gyroscope = sense.get_gyroscope_raw()
        #datos de velocidad rad/s
        gyros_x.append(gyroscope['x'])
        gyros_y.append(gyroscope['y'])
        gyros_z.append(gyroscope['z'])
        
        #t.append(t_actual-t_ini)
        #t_sample.append(t_actual-t_ant)
        
        #t_ant=t_actual

    print("Tenemos 2 segundos muestreados")

    df = pd.DataFrame({'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z, 'gyros_x': gyros_x, 'gyros_y': gyros_y, 'gyros_z': gyros_z})
    #print(df)
    x_test = df.values.reshape(1, 28, 6).astype(input_details["dtype"])
    #x_test = np.expand_dims(x_test, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], x_test)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    print(output)
    prediction = output.argmax()
    print(labels[prediction])
    
    accel_x.clear()
    accel_y.clear()
    accel_z.clear()
    
    gyros_x.clear()
    gyros_y.clear()
    gyros_z.clear()    

