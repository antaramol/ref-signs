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

import keras
from keras.models import Sequential

model_inf = keras.models.load_model("mejor_modelo.h5")

labels = ['anular','ataque','pasos','tecnica']


archivo="ataque" # nombre del archivo
tipo_archivo=".csv" # extensión 

print("grabando")
#bucle for para tomar los datos
while(1):
    for i in range(muestras):
        print(i)
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

    print("Todas las muestras tomadas")

    df = pd.DataFrame({'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z, 'gyros_x': gyros_x, 'gyros_y': gyros_y, 'gyros_z': gyros_z})
    print(df)
    x_test = df.values.reshape(1, 28, 6)
    repuesta = np.argmax(model_inf.predict(x_test))
    print(labels[repuesta])

