import pandas as pd
import numpy as np
import scipy.ndimage
from Gait_UnitStep_V2 import UnitStep


class RawData:

    def __init__(self, path, personId, activity):
        # constants
        self.GAUSS_SIGMA = 20
        self.NUMBER_FEATURES = 28

        # Attributes
        self.path = path
        self.person_id = personId
        self.activity = activity

    # read the data from a file of records
    def read_data(self):
        strId = str(self.person_id).zfill(2)
        file = self.path + "/" + strId + "/" + strId + "_" + str(self.activity).zfill(2) + ".csv"
        try:
            columns = range(1, self.NUMBER_FEATURES + 1)  # skip the first column = timestamp
            data = pd.read_csv(file, sep=",", index_col=None, skiprows=1,
                               usecols=columns).values.astype('int64')
            return (data)

        except FileNotFoundError:
            print("File: ", file, " not found.")
            return (np.array([]))  # No data. Empty arrays

        
    #return the limits according to the type of inflection selected
    #  0 = local minimum of the curve
    #  other = inflection on the ascending curve
    def find_unit_limits(self, data, inflectionType=0):
        if inflectionType==0:
            return(self.find_unit_limits_minimum(data))
        else:
            return(self.find_unit_limits_ascending(data))
                  
    # find the indexes of the boundaries of the unit steps
    def find_unit_limits_minimum(self, data):
        # calculate the derivative for the pressure sensors
        right = int(len(data[0]) / 2)  # half of features for each foot
        lpavg = data[:, 0:8].mean(axis=1)
        rpavg = data[:, right:right + 8].mean(axis=1)
        lpconv = scipy.ndimage.filters.gaussian_filter(lpavg, sigma=self.GAUSS_SIGMA)
        rpconv = scipy.ndimage.filters.gaussian_filter(rpavg, sigma=self.GAUSS_SIGMA)
        diff = [np.diff(lpconv), np.diff(rpconv)]

        # Find the minimum of the curve
        # Process one foot at a time. Left=side 0, right=side 1
        unit_limits = [[], []]
        for side in [0, 1]:
            prev = 0
            for i in range(len(diff[side])):
                if prev < 0 and diff[side][i] >= 0:
                    unit_limits[side].append(i)
                prev = diff[side][i]

        return (unit_limits)

    
    # find the indexes of the boundaries of the unit steps
    def find_unit_limits_ascending(self, data):
        # calculate the derivative for the pressure sensors
        right = int(len(data[0]) / 2)  # half of features for each foot
        lpavg = data[:, 0:8].mean(axis=1)
        rpavg = data[:, right:right + 8].mean(axis=1)
        lpconv = scipy.ndimage.filters.gaussian_filter(lpavg, sigma=self.GAUSS_SIGMA)
        rpconv = scipy.ndimage.filters.gaussian_filter(rpavg, sigma=self.GAUSS_SIGMA)
        diff1 = [np.diff(lpconv), np.diff(rpconv)]
        diff2 = [np.diff(diff1[0]), np.diff(diff1[1])]

        # Find the points of the curve  f’(x)>0 & f’’(x)=0
        # Process one foot at a time. Left=side 0, right=side 1
        unit_limits = [[], []]
        for side in [0, 1]:
            for t in range(len(diff2[side])-1):
                if diff2[side][t]>= 0 and diff2[side][t+1] < 0:
                    unit_limits[side].append(t)
        return (unit_limits)
    
                   
    # Split the data by units
    def split_in_units(self, data, trim=0, minLength=0, inflectionType=0):
        # get the features for each foot
        end = int(len(data[0]) / 2)  # half for each foot
        dataFoot = [data[:, 0:end], data[:, end:2 * end]]

        # get the indexes where the data will be splitted
        limits = self.find_unit_limits(data, inflectionType)
        limit_L = np.concatenate(([0], limits[0], [len(dataFoot[0])]))
        limit_R = np.concatenate(([0], limits[1], [len(dataFoot[1])]))

        # Get the row for each unit. Trim the units of both ends
        unitsList = []
        for i in range(trim, min(len(limit_L), len(limit_R)) - trim - 1):
            left = dataFoot[0][limit_L[i]:limit_L[i + 1]].astype('int64')  # get the rows for the left foot
            right = dataFoot[1][limit_R[i]:limit_R[i + 1]].astype('int64')  # get the rows for the right foot
            if (len(left) >= minLength and len(right) >= minLength):
                unitStep = UnitStep(i, self.person_id, self.activity, sides=[left, right])
                unitsList.append(unitStep)  # add the merged unit
        return (np.array(unitsList))

    # read the data from disk and split it by units.
    def get_list_units(self, trim=0,  minLength=0, inflectionType=0):
        # get the data and split it in units
        data = self.read_data()
        if (len(data) > 0):
            data = self.split_in_units(data, trim,  minLength, inflectionType)
            
        #return the list of units    
        return (data)

