# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:44:47 2023

@author: trist
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseTL(ABC):
    @abstractmethod
    def fit():
        pass
    
    @abstractmethod
    def transform():
        pass
    
    @abstractmethod
    def fit_transform():
        pass


class NormalConditionAlignment(BaseTL):
    """
    Performs Normal Condtion Alignment (NCA) between two the target and source
    domain.
    """
    
    def __init__(self):
        """
        
        """
        pass
 
    def fit(self, Xs, Xt, ys, yt):
        """
        
        """
        self.Xs = Xs
        self.Xt = Xt
        self.ys = ys
        self.yt = yt
        
        self.Xs_standardised = (Xs - np.mean(Xs, axis=0)) / np.std(Xs, axis=0)
        
        # Statistics for the source normal condition data.
        self.Xs_nc = Xs[np.where(ys == 0)[0], :]
        self.Xs_nc_mean = np.mean(self.Xs_nc, axis=0)
        self.Xs_nc_std = np.std(self.Xs_nc, axis=0)
        
        # Statistics for the target normal condition data.
        self.Xt_nc = Xs[np.where(yt == 0)[0], :]
        self.Xt_nc_mean = np.mean(self.Xt_nc, axis=0)
        self.Xt_nc_std = np.std(self.Xt_nc, axis=0)
    
    def transform(self):
        """
        
        """
        Xt_nc_standardised = (self.Xt - self.Xt_nc_mean) / self.Xt_nc_std
        Xt_nc_aligned = (Xt_nc_standardised * self.Xs_nc_std) + self.Xs_nc_mean
        
        return self.Xs_standardised, Xt_nc_aligned
    
    def fit_transform(self, *args):
        """
        
        """
        self.fit(*args)
        return self.transform()