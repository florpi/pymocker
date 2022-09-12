from abc import ABC, abstractmethod
from typing import Optional


class Galaxy(ABC):
    @abstractmethod
    def __init__(self, param):
        self.param = param


class VanillaGalaxy(Galaxy):
    def __init__(
        self,
        log_M_min: Optional[float] = 13.6,
        sigma_log_M: Optional[float] = 0.7,
        kappa: Optional[float] = 0.5,
        log_M1: Optional[float] = 14.4,
        alpha: Optional[float] = 0.92,
    ):
        """Container for the vanilla HOD parameters

        Args:
            log_M_min (Optional[float], optional): _description_. Defaults to 13.6.
            sigma_log_M (Optional[float], optional): _description_. Defaults to 0.7.
            kappa (Optional[float], optional): _description_. Defaults to 0.5.
            log_M1 (Optional[float], optional): _description_. Defaults to 14.4.
            alpha (Optional[float], optional): _description_. Defaults to 0.92.
        """
        self.log_M_min = log_M_min
        self.M_min = 10**self.log_M_min
        self.sigma_log_M = sigma_log_M
        self.kappa = kappa
        self.log_M1 = log_M1
        self.M1 = 10**self.log_M1
        self.alpha = alpha

class AemulusGalaxy(Galaxy):
    def __init__(
        self,
        log_M_min: Optional[float] = 13.6,
        sigma_log_M: Optional[float] = 0.7,
        alpha: Optional[float] = 0.92,
        log_M_sat: Optional[float] = 14.,
        log_M_cut: Optional[float] = 15.,
    ):
        self.log_M_min = log_M_min
        self.M_min = 10**self.log_M_min
        self.sigma_log_M = sigma_log_M
        self.alpha = alpha
        self.log_M_sat = log_M_sat
        self.log_M_cut = log_M_cut
        self.M_sat = 10**log_M_sat
        self.M_cut = 10**log_M_cut

#Eqs 6 and 7 https://iopscience.iop.org/article/10.3847/1538-4357/ab0d7b/pdf
