import tensorflow as tf
import os
import numpy as np
import pandas as pd

#DEveloped classes
import sys
sys.path.append("../libs")
from Gait_Dataset_V2 import Dataset
from ENSEMBLE_CNN_RNN import CNN_LSTM


#*****************************************************************    
# Batch
#*****************************************************************
root = "../../Data"
modelsLSTM = "../../results/LSTM"
modelsCNN = "../../results/CNN"
resultsAVG = "../../results/AVG"

folders = ["7-3", "dropNhalf", "halfNhalf"] #MCCV(30), Sub-MCCV(50), MCCV(50)
Ks=[1,2,3,4]
iterations = [x for x in range(1, 20+1)]

#check if the path for the results exists
if not os.path.exists(resultsAVG): os.mkdir(resultsAVG)
 
#Loop on the data
for i in iterations:
    for k in Ks:
        for folder in folders:        	
            # Creates a session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)

            with session.as_default():
                #Get the path and data
                print("Processing iter", i, " k=",k, " folder=", folder)            
                path = root + "/iter" + str(i) + "/k" + str(k) + "/" + folder

                #Create the CNN_LSTM
                cnn_lstm = CNN_LSTM()
                model_name = "iter" + str(i) + "_k" + str(k) + "_" + folder 

                print("\n\n************************>> Loading the CNN model ... ")
                cnn_lstm.loadCNN(modelsCNN, model_name + ".mdl")

                print("\n\n************************>> Loading the LSTM model ... ")
                cnn_lstm.loadLSTM(modelsLSTM, model_name + ".mdl")

                print("\n\n************************>> Loading the Dataset for testing... ")
                cnn_lstm.loadDataset(path)

                print("\n\n************************>> Predicting ... ")
                cnn_lstm.predict(resultsAVG, i, k, folder, "avg_summary.csv")

            session.close()            
