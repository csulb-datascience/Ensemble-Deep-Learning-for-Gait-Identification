#Libraries Required
import tensorflow as tf
import numpy as np
import pandas as pd

#Developed classes
import sys
sys.path.append("../libs")
from Gait_Dataset_V2 import Dataset

class CNN_LSTM:
        
    #*****************************************************************************************
    #Constructor
    #  Define the attributes for this class
    #*****************************************************************************************
    def __init__(self):
        self.cnn = None
        self.lstm = None
        self.image_shape = (0, 0)
        self.sequence_shape = (0, 0)
        self.classes = []
        self.x = None
        self.y = None
        self.x_splitted = None
        

    #*****************************************************************************************
    # Link the CNN model
    #*****************************************************************************************        
    def linkCNN (self, cnnModel):
        #link the model
        self.cnn = cnnModel
        
        #get the shape of the image
        shp = self.cnn.inputs[0].op.get_attr('shape')
        self.image_shape = (shp.dim[1].size, shp.dim[2].size)

    #*****************************************************************************************
    # Load the CNN model
    #*****************************************************************************************        
    def loadCNN (self, path, model_name):
        #load the model
        self.cnn = tf.keras.models.load_model(path + "/"+ model_name) 
        
        #get the shape of the image
        shp = self.cnn.inputs[0].op.get_attr('shape')
        self.image_shape = (shp.dim[1].size, shp.dim[2].size)
        
    #*****************************************************************************************
    # Load the LSTM model
    #*****************************************************************************************        
    def loadLSTM (self, path, model_name):
        #load the model
        self.lstm = tf.keras.models.load_model(path + "/"+ model_name) 
        
        #get the shape of the sequence
        #shp = self.lstm.inputs[0].op.get_attr('shape')
        #self.sequence_shape = (shp.dim[1].size, shp.dim[2].size)        
            
    #*****************************************************************************************
    # Read the dataset
    #*****************************************************************************************        
    def loadDataset(self, path):
        dataset = Dataset()
        dataset.load_CSV(path, ["test.csv"])
        x, y = dataset.get_sets()
        press_x, _ = dataset.get_sets_single_modal(0)
        acc_x, _ = dataset.get_sets_single_modal(1)
        gyro_x, _ = dataset.get_sets_single_modal(2)        
        self.classes = dataset.get_classes()
    
        #Return the dataset
        self.y = y[0]
        self.x = x[0]
        self.x_splitted = [press_x[0], acc_x[0], gyro_x[0]]

    #*****************************************************************************************
    # Get the most probable Y
    #*****************************************************************************************        
    def getLikelyY(self, predictions):
        likely = max(predictions)
        index = np.where(predictions==likely)[0][0]
        return(self.classes[index], likely)
    
    
    #*****************************************************************************************
    # Predict
    #*****************************************************************************************            
    def predict(self, path, iteration, k, folder, summary_name):
        CNNpredictions = self.cnn.predict(self.x_splitted)
        LSTMpredictions = self.lstm.predict(self.x_splitted)
        counter = np.zeros((3,), dtype=int)        
        
        file_name= "iter" + str(iteration) + "_k" + str(k) + "_" + folder + "_avg.csv"        
        values=[["Y","cnn_y","cnn_hit","lstm_y","lstm_hit","avg_y","avg_hit"]]
        data = pd.DataFrame(data= values)    
        data.to_csv(path + "/" + file_name, header=None, mode="w")
        
        for i in range(len(self.x)):
            AVGpredictions = (CNNpredictions[i] + LSTMpredictions[i]) /2
            cnn_y = cnn_y, cnn_likely = self.getLikelyY(CNNpredictions[i])
            lstm_y, lstm_likely = self.getLikelyY(LSTMpredictions[i])
            avg_y = avg_y, avg_likely = self.getLikelyY(AVGpredictions)            
            
            cnn_hit = (cnn_y == self.y[i])
            lstm_hit = (lstm_y == self.y[i])
            avg_hit = (avg_y == self.y[i])
            
            if (cnn_hit): counter[0] +=1
            if (lstm_hit): counter[1] +=1
            if (avg_hit): counter[2] +=1
                
            #save the summary
            values = [[self.y[i], cnn_y, cnn_hit, lstm_y, lstm_hit,avg_y, avg_hit]]
            data = pd.DataFrame(data= values)    
            data.to_csv(path + "/" + file_name, header=None, mode="a")              
        

        values = [[counter[0] / len(self.x), counter[1] / len(self.x), 
                   counter[2] / len(self.x), iteration, k, folder]]
        data = pd.DataFrame(data= values)    
        data.to_csv(path + "/" + summary_name, header=None, mode="a")              
        
        print("ACCURACY:")
        print("CNN: ", values [0][0])
        print("LSTM: ", values [0][1])
        print("AVG: ", values [0][2])
            
