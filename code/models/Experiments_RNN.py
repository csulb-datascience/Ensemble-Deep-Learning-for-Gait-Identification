import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

#General Libraries
import os
import numpy as np
import pandas as pd

#Developed classes
import sys
sys.path.append("../libs")
from Gait_Dataset_V2 import Dataset


#Read the dataset
def getDataset(path):
    dataset = Dataset()
    dataset.load_CSV(path, ["train.csv", "test.csv"]) 
    classes = dataset.get_classes()

    press_x, y = dataset.get_sets_single_modal(0)
    acc_x, _ = dataset.get_sets_single_modal(1)
    gyro_x, _ = dataset.get_sets_single_modal(2)

    #Convert the labels to a one-hot vector
    y_hot=[]
    for i in range(len(y)):
        y[i] = y[i]-1   #Value of classes in the range from 0 to 13
        y_hot.append(keras.utils.to_categorical(y[i], max(classes)))
    
    x_train = [press_x[0], acc_x[0], gyro_x[0]]
    x_valid = [press_x[1], acc_x[1], gyro_x[1]]

    # Return the dataset
    return (x_train, y_hot[0], x_valid, y_hot[1], max(classes))


def getLSTM(timeSteps, dataDim, num_classes):
    #Build the model as a sequence of layers
    hidden_cells = 64
    model = Sequential()
    model.add(LSTM(hidden_cells, input_shape=(timeSteps, dataDim), return_sequences=True, recurrent_dropout=0.2)) 
    model.add(LSTM(hidden_cells, recurrent_dropout=0.2))    #A second layer of LSTM
    return (model)

def getMultiInputModel(num_classes, timeSteps):
    LSTM_press = getLSTM(timeSteps=timeSteps, dataDim=16, num_classes=num_classes)
    LSTM_acc = getLSTM(timeSteps=timeSteps, dataDim=6, num_classes=num_classes)
    LSTM_gyro = getLSTM(timeSteps=timeSteps, dataDim=6, num_classes=num_classes)

    # Combine the outputs of the LSTMs and complete other layers
    combinedInput = concatenate([LSTM_press.output, LSTM_acc.output, LSTM_gyro.output])
    x = Dense(256, activation="relu")(combinedInput)
    x = Dropout(0.7)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    # Return the final Model
    model = Model(inputs=[LSTM_press.input, LSTM_acc.input, LSTM_gyro.input], outputs=x)
    return(model)


#Save the results in a CSV file
def save_results(path, iteraction, k, folder, summary_name, history):
    #get the values
    epochs = len(history.history["acc"])
    LastAccTrain = history.history["acc"][epochs-1]
    LastAccValid = history.history["val_acc"][epochs-1]    
    maxAccTrain = max(history.history["acc"])
    maxAccValid = max(history.history["val_acc"])
    LastLossTrain = history.history["loss"][epochs-1]
    LastLossValid = history.history["val_loss"][epochs-1]        
    minLossTrain = min(history.history["loss"])
    minLossValid = min(history.history["val_loss"])
    
    #save the summary
    values = [[iteraction, k, folder, epochs, 
              LastAccTrain, LastAccValid, maxAccTrain, maxAccValid,
              LastLossTrain, LastLossValid, minLossTrain, minLossValid]]
    data = pd.DataFrame(data= values)    
    data.to_csv(path + "/" + summary_name, header=None, mode="a")  
    
    #save the history
    header = []
    values = []
    for key, val in history.history.items():
        header.append(key)
        values.append(val)
    saveAs = path + "/iter" + str(iteraction) + "_k" + str(k) + "_" + folder + ".csv"
    data = pd.DataFrame(data= np.transpose(values))
    data.to_csv(saveAs, header=header, mode="w")   
    

    
#parameters
root = "../../Data"
results = "../../results/LSTM/"
folders = ["7-3", "dropNhalf", "halfNhalf"] #MCCV(30), Sub-MCCV(50), MCCV(50)
Ks=[1,2,3,4]
iterations = 20

#check if the path for the results exists
if not os.path.exists(results): os.mkdir(results)
    
#Loop on the data
for i in range(0, iterations):
    for k in Ks:        
        for folder in folders:
            # Creates a session 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)

            with session.as_default():
                #Get the data and the model
                print("Processing iter", i+1, " k=",k, " folder=", folder)            
                path = root + "/iter" + str(i+1) + "/k" + str(k) + "/" + folder

                print("loading the dataset")
                x_train, y_train, x_valid, y_valid, num_classes = getDataset(path)

                print("Getting the model and compiling it")
                model = getMultiInputModel(num_classes=num_classes, timeSteps=len(x_train[0][0]))
                model.compile(loss='categorical_crossentropy',  optimizer=Adam(lr=1e-3), metrics=['accuracy'])  

                #training the model and validating it
                print("Fitting")
                history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), verbose=1)

                #save results
                print("Saving results")
                save_results(results, i+1, k, folder, "summary.csv", history)

                #save the model
                saveAs = results + "/iter" + str(i+1) + "_k" + str(k) + "_" + folder         		    
                model.save(saveAs + ".mdl") 

            session.close()
            