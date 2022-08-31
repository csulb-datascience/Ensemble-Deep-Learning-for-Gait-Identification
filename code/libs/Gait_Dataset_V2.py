import numpy as np
import pandas as pd
from Gait_UnitStepList_V2 import UnitStepList


class Dataset:    
    
    def __init__(self):
      #Attributes
        self.data = UnitStepList()
        self.sublist = []

    #read the data from files
    def read_units(self, path, participants, activities, trim=2,  minLength=0, inflectionType=0, filtered=False):        
        #read the data for each participant and activity
        units = UnitStepList()
        for personId in participants:
            for activity in activities:
                tempList = UnitStepList()
                tempList.from_raw_data(path, personId, activity, trim,  minLength, inflectionType)
                if filtered: tempList.filterOutliers()
                units.append(tempList)        
        return(units)
        
                       
    #read the data from files
    def read_dataset(self, path, participants, activities, trim=2,  minLength=0, 
                     fixedSize=None, inflectionType=0, filtered=False, allPositive=False):
        #read the data for each participant and activity
        #Merge and resize units as the smallest one. 
        #normalize all the units as a unique list
        units = self.read_units(path, participants, activities, trim,  minLength, inflectionType, filtered)           
        unitSize = 0
        if(units.length()>0):
            unitSize = units.smallest() if fixedSize == None else fixedSize
            print("Smallest unit = ", units.smallest())
            print("Resizing to ", unitSize)
            units.merge_sides_units(unitSize)        
            units.normalize(allPositive) 
        #assign the results
        self.data = units
        self.sublist=[]
        self.sublist.append(self.data)
        return(unitSize)
                                              
    #create sublists of the data according to a passed distribution
    def get_sublists(self, n=0, distribution=[100], k=1, shuffle=True):
        groups = self.data.get_groups(k)
        if shuffle: groups.shuffle()
        if (n==0 ): n=groups.number_units()
        self.sublist = groups.get_sublists(n, distribution)
                                                          
    #save the sublists on disk
    #SaveAs: is a list of names for each sublist
    def save_CSV(self, path, saveAs):         
        for i in range(len(saveAs)):
            self.sublist[i].save_list(path, saveAs[i])     

    #load the data from a file on disk
    def load_CSV(self, path, file_name_list):
        sublist = []
        for i in range(len(file_name_list)):
            sublist.append(UnitStepList())
            sublist[i].load_list(path, file_name_list[i])
        self.sublist = np.array(sublist)
                
    #return the classes included in the units
    def get_classes(self):
        classes=[]
        for i in range(len(self.sublist)):
            for j in range(self.sublist[i].number_units()):
                if(self.sublist[i].units[j].person_id not in classes):
                    classes.append(self.sublist[i].units[j].person_id)
        return(np.arange(1,int(max(classes))+1))
        #return(np.sort(np.array(classes).astype("int")))
    
    #return the sets in a format for training and testing
    def get_sets(self):
        #room for the features and labels
        x = [0] * len(self.sublist)
        y = [0] * len(self.sublist)        
        
        #get the features and labels for each sublist
        for i in range(len(self.sublist)):
            x[i] = np.array([self.sublist[i].units[0].rows])
            y[i] = np.array([self.sublist[i].units[0].person_id])
            for j in range(1, self.sublist[i].number_units()):
                x[i] = np.concatenate((x[i], [self.sublist[i].units[j].rows]))
                y[i] = np.concatenate((y[i], [self.sublist[i].units[j].person_id]))  
                
        #return the sets
        return(np.array(x), np.array(y))
            
    #return the sets of one component in a format for training and testing
    # item:  0=pressure, 1=accelerometer, 2=gyro     
    def get_sets_single_modal(self, item):
        #ranges
        left = [range(0,8), range(8,11), range(11,14)]
        right= [range(14,22), range(22,25), range(25,28)]
        
        #room for the features and labels
        x = [0] * len(self.sublist)
        y = [0] * len(self.sublist)        
        
        #get the features and labels for each sublist
        for i in range(len(self.sublist)):
            rows = np.concatenate((self.sublist[i].units[0].rows[:,left[item]], 
                                   self.sublist[i].units[0].rows[:,right[item]]), axis=1)

            x[i] = np.array([rows])
            y[i] = np.array([self.sublist[i].units[0].person_id])
            for j in range(1, self.sublist[i].number_units()):
                rows = np.concatenate((self.sublist[i].units[j].rows[:,left[item]], 
                                       self.sublist[i].units[j].rows[:,right[item]]), axis=1)
                x[i] = np.concatenate((x[i], [rows]))
                y[i] = np.concatenate((y[i], [self.sublist[i].units[j].person_id]))  
                
        #return the sets
        return(np.array(x), np.array(y))
    
    
    #return the sets of two components in a format for training and testing
    # item:  0=pressure, 1=accelerometer, 2=gyro     
    def get_sets_dual_modal(self, item1, item2):
        #ranges
        left = [range(0,8), range(8,11), range(11,14)]
        right= [range(14,22), range(22,25), range(25,28)]
        
        #room for the features and labels
        x = [0] * len(self.sublist)
        y = [0] * len(self.sublist)        
        
        #get the features and labels for each sublist
        for i in range(len(self.sublist)):
            rows = np.concatenate((self.sublist[i].units[0].rows[:,left[item1]], 
                                   self.sublist[i].units[0].rows[:,left[item2]], 
                                   self.sublist[i].units[0].rows[:,right[item1]],
                                   self.sublist[i].units[0].rows[:,right[item2]]), axis=1)
            x[i] = np.array([rows])
            y[i] = np.array([self.sublist[i].units[0].person_id])
            for j in range(1, self.sublist[i].number_units()):
                rows = np.concatenate((self.sublist[i].units[j].rows[:,left[item1]], 
                                       self.sublist[i].units[j].rows[:,left[item2]], 
                                       self.sublist[i].units[j].rows[:,right[item1]],
                                       self.sublist[i].units[j].rows[:,right[item2]]), axis=1)
                x[i] = np.concatenate((x[i], [rows]))
                y[i] = np.concatenate((y[i], [self.sublist[i].units[j].person_id]))  
                
        #return the sets
        return(np.array(x), np.array(y))


       # return the sets of one component in a format for training and testing
        # item:  0=pressure, 1=accelerometer, 2=gyro
    def get_sets_single_modal_by_row(self, item, numRows=1):
        # ranges
        left = [range(0, 8), range(8, 11), range(11, 14)]
        right = [range(14, 22), range(22, 25), range(25, 28)]

        # room for the features and labels
        x = [0] * len(self.sublist)
        y = [0] * len(self.sublist)

        # get the features and labels for each sublist
        for i in range(len(self.sublist)):
            rows = np.concatenate((self.sublist[i].units[0].rows[0:numRows, left[item]],
                                   self.sublist[i].units[0].rows[0:numRows, right[item]]), axis=1)
            x[i] = np.array([rows])
            y[i] = np.array([self.sublist[i].units[0].person_id])
            
            for j in range(0, self.sublist[i].number_units()):
                rows = np.concatenate((self.sublist[i].units[j].rows[:, left[item]],
                                       self.sublist[i].units[j].rows[:, right[item]]), axis=1)
                for k in range(0, len(rows), numRows):
                    if k+numRows <= len(rows):
                        #group = np.array([rows[k: k+numRows, :]])
                        x[i] = np.concatenate((x[i], [rows[k: k+numRows, :]]))
                        y[i] = np.concatenate((y[i], [self.sublist[i].units[j].person_id]))

                # return the sets
        return (np.array(x)[:,1:], np.array(y)[:,1:])