
import numpy as np

class PointsInfoStruct:
    def __init__(self, points, nPoints, nDimensions, nClusters=None, assignments=None ):
        self.points = np.array(points)
        self.nPoints = nPoints
        self.nDimensions = nDimensions
        self.nClusters = nClusters
        self.assignments = np.array(assignments)


    def reset_points(self, new_points, new_assignments=None, new_nClusters=1):
        assert len(new_points) == self.nPoints
        self.points = np.array(new_points)
        self.assignments = [] if new_assignments is None else new_assignments
        self.assignments = np.array(self.assignments)
        for pt in new_points:
            assert len(pt) == self.nDimensions
        self.nClusters = new_nClusters



def extr(line):
    return line.strip("\n ").split(" ")

class UnivariateData(PointsInfoStruct):
   def __init__(self):

       self.assignments = []
       self.points = []
       self.nClusters = 1

       print("Loading data from data_single.in ... \n")
       with open("model/data_single.in", "r") as fd:
           self.nPoints, self.nDimensions = extr(fd.readline())
           for i in range(self.nPoints):
               point = extr(fd.readline())
               self.points.append(point)

       self.points = np.array(self.points)


class MultivariateData(PointsInfoStruct):
    def __init__(self):

        self.assignments = []
        self.points = []

        with open("model/data_multiple.in", "r") as fd:
            self.nPoints, self.nDimensions, self.nClusters = extr(fd.readline())
            for i in range(self.nPoints):
                linecontents = extr(fd.readline())
                assert len(linecontents) == self.nDimensions + 1
                self.points.append(linecontents[:-1])
                self.assignments.append(linecontents[-1])

        self.points = np.array(self.points)
        self.assignments = np.array(self.assignments)


