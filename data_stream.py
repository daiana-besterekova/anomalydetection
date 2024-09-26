import numpy as np
import pandas as pd

def stream_data(timestamps, trend_factor, seasonal_amplitude1, seasonal_amplitude2, frequency1, frequency2, noise_level, random_spike_prob):
    '''
    Generates a simulated data stream with trend, seasonal components, random noise, and occasional random spikes.
    
    Parameters:
    - timestamps: Array of timestamps to iterate over.
    - trend_factor: Factor that controls the rate of trend change over time.
    - seasonal_amplitude1: Amplitude of the first seasonal component.
    - seasonal_amplitude2: Amplitude of the second seasonal component.
    - frequency1: Frequency of the first seasonal component.
    - frequency2: Frequency of the second seasonal component.
    - noise_level: Standard deviation of the normal noise added to the data points.
    - random_spike_prob: Probability of generating a random spike in the data.
    
    Yields:
    - (timestamp, data_point): A tuple where `timestamp` is the current time point and `data_point` is the generated value.
    '''
    
    trend = 0
    for t in timestamps:
        # Add randomness to trend 
        trend += trend_factor * np.random.uniform(-1, 1) * t
        
        # Seasonal components with varying amplitude
        seasonal1 = (seasonal_amplitude1 + np.random.uniform(-1, 1)) * np.sin(2 * np.pi * t / frequency1)
        seasonal2 = (seasonal_amplitude2 + np.random.uniform(-0.5, 0.5)) * np.sin(2 * np.pi * t / frequency2)
        
        # Random noise with varying amplitude
        noise = np.random.normal(0, noise_level * np.random.uniform(0.5, 1.5))
        
        # Randomly decide whether to add a spike to the data point.
        # There's a probability of `random_spike_prob` to generate a spike (value 1), 
        # and a probability of `1 - random_spike_prob` to have no spike (value 0).
        random_spike = np.random.choice([0, 1], p=[1 - random_spike_prob, random_spike_prob])
        spike_magnitude = np.random.uniform(-5, 5) if random_spike else 0
        
        # Data point is the sum of trend, seasonal components, noise, and possible spike
        data_point = trend + seasonal1 + seasonal2 + noise + spike_magnitude
        
        # Yield timestamp and data point as a tuple
        yield pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(t)), data_point
