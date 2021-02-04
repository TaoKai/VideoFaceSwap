import numpy as np

class KalmanObject:
    def __init__(self, m, Qval, Rval):
        self.K = np.zeros((m, m))
        self.xx = np.zeros(m)
        self.P = np.eye(m)
        self.F = np.eye(m)
        self.B = np.eye(m)
        self.H = np.eye(m)
        self.Q = Qval * np.eye(m)
        self.R = Rval * np.eye(m)

    def kalman_update(self, uu, zz):
        self.xx = self.F.dot(self.xx) + self.B.dot(uu)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.R))
        self.xx = self.xx + self.K.dot(zz - self.H.dot(self.xx))
        self.P = self.P - self.K.dot(self.H).dot(self.P)