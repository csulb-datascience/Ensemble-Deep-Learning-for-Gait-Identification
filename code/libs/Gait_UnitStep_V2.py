import scipy.ndimage
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class UnitStep:
    # Define attributes
    def __init__(self, unitId=0, personId=0, activity=0, rows=np.empty((0, 0)), sides=np.empty((2, 0, 0))):
        # Attributes
        self.unit_id = unitId
        self.person_id = personId
        self.activity = activity
        self.rows = np.array(rows)
        self.sides = np.array(sides)

    # return the shape of the units
    def shape(self):
        return (self.height(), self.width())

    # return the number of rows of the unit
    def height(self):
        # get the number of rows or the minimum side
        x, _ = self.rows.shape
        if (x == 0):
            i = 0 if len(self.sides[0]) < len(self.sides[1]) else 1
            x = len(self.sides[i])
        return (x)

    # return the number of features
    def width(self):
        _, x = self.rows.shape
        return (x)

    # True if both units belong to the same person and activity
    def same_group(self, unit):
        return (self.person_id == unit.person_id and self.activity == unit.activity)

    # Return the unit as a plain array
    def flatten(self):
        unitId = np.transpose([[self.unit_id] * self.height()])
        personId = np.transpose([[self.person_id] * self.height()])
        activity = np.transpose([[self.activity] * self.height()])
        flat = np.concatenate((personId, activity, unitId, self.rows), axis=1)
        return (flat)

    # Converts a plain array as a unit
    def unflatten(self, flat):
        self.person_id = flat[0, 0]
        self.activity = flat[0, 1]
        self.unit_id = flat[0, 2]
        self.rows = flat[:, 3:]

    # Merge the Left and right sides in one group of rows
    def merge_sides(self, targetSize):
        # Resize each side and concatenate them
        left = scipy.ndimage.zoom(self.sides[0], [targetSize / len(self.sides[0]), 1])
        right = scipy.ndimage.zoom(self.sides[1], [targetSize / len(self.sides[1]), 1])
        self.rows = np.concatenate((left, right), axis=1)
        self.sides = np.empty((2, 0, 0))  # delete the sides
