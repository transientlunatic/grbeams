import numpy as np
from scipy import stats

class Distribution():
    """
    Represents a statistical distribution.
    
    """
    pass

class DeltaDistribution(Distribution):
    """
    Represents a delta distribution.
    """
    name = "delta"
    ndim = 1
    def __init__(self, x):
        """
        A dirac delta distribution.
        
        Parameters
        ----------
        x : float
            The location of the delta spike.
        """
        self.x = x
        self.range = x
        pass
    
    def pdf(self, e):
        """
        Calculate the probability density at a given value.
        
        Parameters
        ----------
        e : float
            The value at which the PDF should be evaluated.
        """
        if e == self.x:
            return 1.0
        else:
            return 0.0

class JeffreyDistribution(Distribution):
    """
    Represents a Jeffrey's distribution.
    """
    name = "jeffrey"
    ndim = 2
    range = (0,1.0)
    def __init__(self):
        """
        A Jeffrey distribution.
        
        Parameters
        ----------
        x : float
            The location of the delta spike.
        """
        pass
    
    def pdf(self, e):
        prior_dist = stats.beta(0.5,0.5)
        return prior_dist.pdf(e)

class UniformDistribution(Distribution):
    """
    Represents a uniform distribution.
    """
    name = "uniform"
    ndim = 2
    def __init__(self, range=(0.0,1.0)):
        """
        A uniform distribution.
        
        Parameters
        ----------
        range : tuple
           The range over which the uniform distribution should be evaluated. 
           This can either be a tuple of the start and end values, or an array 
           which spans the values.
        """
        self.range = np.array(range)
        pass
    
    def pdf(self, e):
        return ((e>min(self.range)) & (e<=max(self.range))) * 1./(max(self.range)-min(self.range))
