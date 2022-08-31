import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from Gait_RawData_V2 import RawData
from Gait_UnitStep_V2 import UnitStep


class UnitStepList:    
    
    #Constructor
    def __init__(self, units=np.empty((0, ))):        
        #Attributes
        self.units = np.array(units)

    #return the length of the list
    def length(self):
        return(len(self.units))

    #return the width of the list
    def width(self):
        if(len(self.units) > 0):
            return(self.units[0].width())
        return(0)
    
    #return the number of units
    def number_units(self):
        return(len(self.units))
    
    #return a list of person ids and their number of units 
    def units_per_id(self):
        counter = dict()
        for unit in self.units:
            if unit.person_id not in counter: counter[unit.person_id]=0
            counter[unit.person_id] = counter[unit.person_id] + 1             
            
        summary = np.array([[k,v] for k,v in counter.items()])
        return(summary)
    
    #return the smallest height of units
    def smallest(self):
        return(np.array([self.units[i].height() for i in range(self.number_units())]).min())

    #return the average height of units
    def average(self):
        return(np.average([self.units[i].height() for i in range(self.number_units())]))

    #return the average height of units
    def averageLeftRight(self):
        left = [len(self.units[i].sides[0]) for i in range(self.number_units())]
        right = [len(self.units[i].sides[0]) for i in range(self.number_units())]        
        return(np.average(left + right))
    
    #Append units to the list
    def append(self, listUnits):
        self.units = np.concatenate((self.units, listUnits.units))
    
    #Shuffle the list
    def shuffle(self):
        np.random.shuffle(self.units) 
    
    #merge sides and resize units in the list to the smallest size
    def merge_sides_units(self, targetSize):
        for i in range(self.number_units()):
            self.units[i].merge_sides(targetSize)    
    
    #group the data
    def group(self, k):
        self.units = self.get_groups(k).units
    
    #group by k units. Assumes that the unit's sides were merged
    def get_groups(self, k):
        groupList = []
        i=0
        while(i <= self.number_units()-k):
            unit = copy.deepcopy(self.units[i])
            j=1
            while(j<k and unit.same_group(self.units[i+j])):
                unit.rows= np.concatenate((unit.rows, self.units[i+j].rows))
                j +=1                
            #if completed the group insert it in the list
            if(j==k): groupList.append(unit)
            i=i+j
        #return the grouped units
        return(UnitStepList(np.array(groupList)))
    
    #Add the list of units from a file
    def from_raw_data(self, path, personId, activity, trim=0, minLength=0, inflectionType=0):
        rawData = RawData(path, personId, activity)
        listUnits = rawData.get_list_units(trim,  minLength, inflectionType)            
        self.units = np.concatenate((self.units, listUnits))        
    
    #filter outliers
    def filterOutliers(self):
        if self.number_units() > 0:
            #get the heights for analysis
            heights =[]
            for i in range(self.number_units()):
                heights.append([self.units[i].height(), i])
            heights = np.array(heights)

            #Filter the outliers 
            df = pd.DataFrame(data=heights, columns=["height","idx"])
            Q1 = df['height'].quantile(0.25)
            Q3 = df['height'].quantile(0.75)
            IQR = Q3 - Q1
            filtered = df.query('(@Q1 - 1.5 * @IQR) <= height <= (@Q3 + 1.5 * @IQR)')
        
            #Filter the units
            filteredUnits = []
            for i in filtered['idx']:
                filteredUnits.append(self.units[i])            
            self.units = np.array(filteredUnits)
        
    
    #return lists of units splitted according to the percentage
    def get_sublists(self, n, distribution):
        distribution = np.array(distribution)*n/100.0
        index = np.append(0, np.cumsum(distribution)).astype('int')            
        sublist=[] 
        for i in range(len(distribution)):
            sublist.append(UnitStepList(self.units[index[i] : index[i+1]]))            
        #return the sublists
        return(np.array(sublist))
                        
    #flatten the list of units
    def flatten(self):
        flat= self.units[0].flatten() if (self.length()>0) else []
        for i in range(1, self.length()):
            flat = np.concatenate((flat, self.units[i].flatten()))                
        return(flat)
        
    #unflatten from a plain array
    def unflatten(self, flat, unitHeight):
        unitsList = []
        for i in range(0, (len(flat) // unitHeight) * unitHeight, unitHeight):
            rows = flat[i: i+unitHeight]
            unit = UnitStep()
            unit.unflatten(rows)
            unitsList.append(unit)            
        #reference to the attribute of the class
        self.units=np.array(unitsList)
        
   #Normalize the values between (0,1) for pressure and (-1,1) for acc and gyro
    def normalize(self, allPositive=False):        
        sensorRange=np.array( [ [[0.],[2.]], [[-32768.],[32768.]] ])
        rangeId=[0,0,0,0,0,0,0,0,1,1,1,1,1,1] * 2
        minValue = [0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1] * 2 
        if allPositive: minValue = [0,0,0,0,0,0,0] * 4
        
        data = self.flatten()                    #Convert the units in a 2D array
        extraCols = len(data[0]) - self.width()  #extra columns are not features
        normalized = []
        for i in range(len(data[0])):            #Process each column of the data
            after = data[:, i]
            if(i >= extraCols):  #normalize only the features
                after = after.reshape(-1,1).astype("float64")

                #********************************************************************
                #Temporary to create an special dataset
                #if minValue[i-extraCols] == -1:
                #    for k in range(len(after)):
                #        if (after [k] == 0.0): after [k]=0.01
                #********************************************************************

                scaler = MinMaxScaler(feature_range=(minValue[i-extraCols], 1))
                scaler.fit(sensorRange[rangeId[i-extraCols]])
                #scaler.fit(after)
                after = scaler.transform(after)[:,0]            

                #********************************************************************
                #Temporary to create an special dataset
                #if minValue[i-extraCols] == 0:
                #    for k in range(len(after)):
                #        if (after [k] == 0.0): after [k]=0.01
                #********************************************************************

            normalized.append(after)             
        #convert the normalized data in a list of units
        self.unflatten(np.transpose(normalized), self.units[0].height())
    
    #save the data in a file on disk
    def save_list(self, path, saveAs, mode='w'): 
        header=["user_id","activity","unit_id",
               "lp1","lp2","lp3","lp4","lp5","lp6","lp7","lp8","lax","lay","laz","lgx","lgy","lgz",
               "rp1","rp2","rp3","rp4","rp5","rp6","rp7","rp8","rax","ray","raz","rgx","rgy","rgz"]
        flat = self.flatten()                           
        data = pd.DataFrame(data=flat)
        data.to_csv(path+"/"+saveAs, header=header, mode=mode) 
        
    #load data saved as a flat array
    def load_list(self, path, file_name, numCols=32):
        #read the fie but skip the first column = seqId
        columns = range(1, numCols) 
        data = pd.read_csv(path+"/"+file_name, sep=",", index_col=None, 
                           usecols = columns).values.astype('float64')
        
        #get the height of the first unit
        i = 0
        while(i<len(data) and data[i][0]==data[i+1][0] 
            and data[i][1]==data[i+1][1] and data[i][2]==data[i+1][2]): i+=1
        
        #recover the units assuming the height of the firt unit
        unitHeight = i+1
        self.unflatten(data, unitHeight)
        
        