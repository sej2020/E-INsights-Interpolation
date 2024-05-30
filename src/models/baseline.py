"""Implements linear interpolation baseline model.

Typical usage example:
```python
>>> from src.models.baseline import LinearInterpolation
>>> x = numpy.array([1, 2, 3, 4, 6, 7])
>>> y = numpy.array([20, 19, 18, 17, 11, 10])
>>> model = LinearInterpolation()
>>> model.fit(x, y)
>>> print(model.predict(np.array([5]), ablation_start=4)
14.
>>> complete_x, complete_y = model.interpolate()
>>> print(complete_x)
array([1, 2, 3, 4, 5, 6, 7])
>>> print(complete_y)
array([20, 19, 18, 17, 14, 11, 10])
```
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple

class LinearInterpolation:
    """
    Performs linear interpolation on time-series data 

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
    """
    def __init__(self):
        """
        Initializes an instance of the LinearInterpolation class.
        """
        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits model to data. Modifies object attributes and returns nothing.

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        self.x = x
        self.y = y

    def _point_predict(self, x: float) -> float:
        """
        Predicts y value for x.

        Args:
            x: x value.

        Returns:
            y value.

        Raises:
            Exception: if model is not fitted.
        """
        if self.x is None or self.y is None:
            raise Exception("Model not fitted.")

        if x < self.x[0]:
            return self.y[0]
        elif x > self.x[-1]:
            return self.y[-1]
        else:
            i = np.searchsorted(self.x, x)
            x0 = self.x[i - 1]
            x1 = self.x[i]
            y0 = self.y[i - 1]
            y1 = self.y[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        
    def predict(self, x: np.ndarray, ablation_start: int = None, units: str = "s") -> np.ndarray:
        """
        Predicts y values for x values.

        Args:
            x: x values.
            ablation_start: index of where the first missing value would be placed in the array fitted to x. For example, if the array 
                fitted to x is [3,4,6,7], the ablation start index should be 2, because the missing value would be in position 2 if the 
                array was uninterrupted. Note: if this parameter is None, function will have to predict missing values in linear time 
                instead of constant time.
            units: unit of time for the x values. Default is 's' for seconds (unused).

        Returns:
            predicted y values.

        Raises:
            Exception: if model is not fitted.
        """
        if self.x is None or self.y is None:
            raise Exception("Model not fitted.")
        
        if ablation_start is None:
            warnings.warn(
                "No ablation start specified, predicting missing values will be in linear time instead of constant time. Expect long runtimes."
                )
            return np.array([self._point_predict(x_i) for x_i in x])
        
        else:
            pre_gap_value_x = self.x[ablation_start - 1]
            pre_gap_value_y = self.y[ablation_start - 1]
            post_gap_value_x = self.x[ablation_start]
            post_gap_value_y = self.y[ablation_start]
            slope = (post_gap_value_y - pre_gap_value_y) / (post_gap_value_x - pre_gap_value_x)
            predictions = np.arange(1, len(x)+1) * slope
            predictions += pre_gap_value_y
            return predictions
        
    
    def interpolate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates data.

        Returns:
            x values with missing interval interpolated, y values with missing interval interpolated.

        Raises:
            Exception: if model is not fitted.

        Notes:
            Needs to be improved to run in constant time if missing data location is known. Currently runs in linear time.
        """
        if self.x is None or self.y is None:
            raise Exception("Model not fitted.")

        complete_x = np.arange(self.x[0], self.x[-1] + 1)
        complete_y = np.array([self._point_predict(x) for x in complete_x])
        return complete_x, complete_y
    

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 14, 15, 16, 17, 19, 20])
    y = np.array([145, 110, 109, 118, 161, 150, 138, 140, 112, 126, 98, 159, 37, 45, 180])
    model = LinearInterpolation()
    model.fit(x, y)
    print(model.predict(np.array([18]), ablation_start=13))
    complete_x, complete_y = model.interpolate()
    print(complete_x)
    print(complete_y)
    plt.plot(complete_x, complete_y)
    plt.show()