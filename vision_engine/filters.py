"""
NETRAX AI - Kalman Filters
Smoothing filters for ultra-stable tracking
"""

import numpy as np
from typing import Tuple

class KalmanFilter:
    """
    1D Kalman filter for smoothing landmark coordinates
    Reduces jitter and noise in tracking data
    """
    
    def __init__(self, process_noise: float = 0.01, 
                 measurement_noise: float = 0.1):
        """
        Initialize Kalman filter
        
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        # State estimate
        self.x = 0.0  # Position
        self.v = 0.0  # Velocity
        
        # State covariance
        self.P = np.array([[1.0, 0.0],
                          [0.0, 1.0]])
        
        # Process noise covariance
        self.Q = np.array([[process_noise, 0.0],
                          [0.0, process_noise]])
        
        # Measurement noise covariance
        self.R = measurement_noise
        
        # State transition matrix
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])
        
        # Measurement matrix
        self.H = np.array([[1.0, 0.0]])
        
        self.initialized = False
    
    def update(self, measurement: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Update filter with new measurement
        
        Args:
            measurement: New measurement value
            dt: Time delta (default 1.0)
            
        Returns:
            Tuple of (filtered_position, filtered_velocity)
        """
        # Initialize on first measurement
        if not self.initialized:
            self.x = measurement
            self.v = 0.0
            self.initialized = True
            return self.x, self.v
        
        # Update state transition matrix with dt
        self.F[0, 1] = dt
        
        # Prediction step
        # x_pred = F * x
        state = np.array([[self.x], [self.v]])
        state_pred = self.F @ state
        
        # P_pred = F * P * F' + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        # y = z - H * x_pred (innovation)
        innovation = measurement - (self.H @ state_pred)[0, 0]
        
        # S = H * P_pred * H' + R (innovation covariance)
        S = (self.H @ P_pred @ self.H.T)[0, 0] + self.R
        
        # K = P_pred * H' * S^-1 (Kalman gain)
        K = (P_pred @ self.H.T) / S
        
        # x = x_pred + K * y
        state = state_pred + K * innovation
        
        # P = (I - K * H) * P_pred
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred
        
        # Update internal state
        self.x = state[0, 0]
        self.v = state[1, 0]
        
        return self.x, self.v
    
    def reset(self):
        """Reset filter to initial state"""
        self.x = 0.0
        self.v = 0.0
        self.P = np.array([[1.0, 0.0],
                          [0.0, 1.0]])
        self.initialized = False


class KalmanFilter2D:
    """
    2D Kalman filter for smoothing x,y coordinates
    """
    
    def __init__(self, process_noise: float = 0.01, 
                 measurement_noise: float = 0.1):
        self.filter_x = KalmanFilter(process_noise, measurement_noise)
        self.filter_y = KalmanFilter(process_noise, measurement_noise)
    
    def update(self, x: float, y: float, dt: float = 1.0) -> Tuple[float, float]:
        """Update with 2D measurement"""
        filtered_x, _ = self.filter_x.update(x, dt)
        filtered_y, _ = self.filter_y.update(y, dt)
        return filtered_x, filtered_y
    
    def reset(self):
        """Reset both filters"""
        self.filter_x.reset()
        self.filter_y.reset()