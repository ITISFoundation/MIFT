class Param:
    """A class for handling deformation data.

    Arguments:
        - name: name of the data file
        - path_data: path to the data file

    Attributes:
        - N: number of time points
        - dAP: resolution (in m) in AP direction
        - dRL: resolution (in m) in RL direction
        - dFH: resolution (in m) in FH direction
        - nAP: number of points (in space ) in AP direction
        - nRL: number of points (in space ) in RL direction
        - nFH: number of points (in space ) in FH direction
    """

    def __init__(self, name, path_data):
        self.name = name
        self.path_data = path_data
        self.N = 20
        self.dAP = 0.93e-3
        self.dRL = 1e-3
        self.dFH = 0.93e-3
        self.nAP = 320
        self.nRL = 190
        self.nFH = 320

    def GetData(self, f):
        self.LPH_dispAPcorr=f['LPH_dispAPcorr']
        self.LPH_dispRLcorr=f['LPH_dispRLcorr']
        self.LPH_dispFHcorr=f['LPH_dispFHcorr']
        self.DencRL = f['DencRL']
        self.DencFH = f['DencFH']
        self.DencAP = f['DencAP']      
        self.GM =	f['GM']
        self.WM =	f['WM']
        self.CSF =	f['CSF']


class Axis:
    """a class for handling cartesian coordinate system"""

    def __init__(self, x, y, z):
        self.XAxis = x
        self.YAxis = y
        self.ZAxis = z


class Deformation:
    """
    a class for handling all three components of the deformation data
    """

    def __init__(self, ux, uy, uz):
        self.ux = ux
        self.uy = uy
        self.uz = uz


def PutValueId(id, deformation, value):
    """
    a function that get "deformation" (type Deformation), id , and a value
    and replace the ids in all three components of the deformation with value
    """
    temp = deformation.ux[:, :, :]
    temp[id] = value
    deformation.ux[:, :, :] = temp

    temp = deformation.uy[:, :, :]
    temp[id] = value
    deformation.uy[:, :, :] = temp

    temp = deformation.uz[:, :, :]
    temp[id] = value
    deformation.uz[:, :, :] = temp

    return deformation
