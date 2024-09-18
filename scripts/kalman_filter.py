from lds.inference import filterLDS_SS_withMissingValues_np, smoothLDS_SS
import numpy as np

class KalmanFilter():

    def __init__(self,
                    pos_x0: float = 0.0,
                    pos_y0: float = 0.0,
                    vel_x0: float = 0.0,
                    vel_y0: float = 0.0,
                    acc_x0: float = 0.0,
                    acc_y0: float = 0.0,
                    sigma_a: float = 1000.0,
                    sigma_x: float = 100.0,
                    sigma_y: float = 100.0,
                    sqrt_diag_V0_value: float = 1e-03,
                    fps: int = 60
                    ) -> None:
        
        self.pos_x0=pos_x0
        self.pos_y0=pos_y0
        self.vel_x0=vel_x0
        self.vel_y0=vel_y0
        self.acc_x0=acc_x0
        self.acc_y0=acc_y0
        self.sigma_a=sigma_a
        self.sigma_x=sigma_x
        self.sigma_y=sigma_y
        self.sqrt_diag_V0_value=sqrt_diag_V0_value
        self.fps=fps

        if np.isnan(self.pos_x0):
            self.pos_x0 = 0

        if np.isnan(self.pos_y0):
            self.pos_y0 = 0

        self.dt = 1.0 / self.fps

        self.B = np.array([  [1,     self.dt,    0.5*self.dt**2, 0,      0,          0],
                        [0,     1,          self.dt,        0,      0,          0],
                        [0,     0,          1,              0,      0,          0],
                        [0,     0,          0,              1,      self.dt,    0.5*self.dt**2],
                        [0,     0,          0,              0,      1,          self.dt],
                        [0,     0,          0,              0,      0,          1]],
                      dtype=np.double)
        
        self.Z = np.array([  [1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0]],

                      dtype=np.double)

        self.Qe = np.array([    [self.dt**4/4,  self.dt**3/2,   self.dt**2/2,   0,              0,              0],
                                [self.dt**3/2,  self.dt**2,     self.dt,        0,              0,              0],
                                [self.dt**2/2,  self.dt,        1,              0,              0,              0],
                                [0,             0,              0,              self.dt**4/4,   self.dt**3/2,   self.dt**2/2],
                                [0,             0,              0,              self.dt**3/2,   self.dt**2,     self.dt],
                                [0,             0,              0,              self.dt**2/2,   self.dt,        1]],
                                dtype=np.double)

        self.R = np.diag([self.sigma_x**2, self.sigma_y**2]).astype(np.double)
        self.m0 = np.array([self.pos_x0, self.vel_x0, self.acc_x0, self.pos_y0, self.vel_y0, self.acc_y0], dtype=np.double)
        self.V0 = np.diag(np.ones(len(self.m0))*self.sqrt_diag_V0_value**2).astype(np.double)
        self.Q = self.Qe * self.sigma_a

    def filter(self, y):
        self.filtered_results = filterLDS_SS_withMissingValues_np(
            y=y, B=self.B, Q=self.Q, m0=self.m0, V0=self.V0, Z=self.Z, R=self.R)
        means = self.filtered_results["xnn"]
        covs = self.filtered_results["Vnn"]
        std_devs = np.sqrt(np.diagonal(covs, axis1=0, axis2=1))
        return means, std_devs

    def smooth(self, y):
        filtered_means, filtered_std_devs = self.filter(y)
        self.smoothed_results = smoothLDS_SS(
            B=self.B, xnn=self.filtered_results["xnn"], Vnn=self.filtered_results["Vnn"],
            xnn1=self.filtered_results["xnn1"], Vnn1=self.filtered_results["Vnn1"], m0=self.m0, V0=self.V0)
        means = self.smoothed_results["xnN"]
        covs = self.smoothed_results["VnN"]
        std_devs = np.sqrt(np.diagonal(covs, axis1=0, axis2=1))
        return means, std_devs
