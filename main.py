import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from data_stream import stream_data
from anomaly_detection import detect_first_layer_anomalies, detect_neighbor_deviation, validate_with_isolation_forest

# Parameters for simulation
n_points = 1000  # Number of data points to simulate
trend_factor = 0.0002  # Trend factor for adding a gradual change in the data
seasonal_amplitude1 = 3  # Amplitude of the first seasonal component
seasonal_amplitude2 = 2  # Amplitude of the second seasonal component
frequency1 = 100  # Frequency of the first seasonal component
frequency2 = 150  # Frequency of the second seasonal component
noise_level = 0.3  # Standard deviation of the normal noise
random_spike_prob = 0.05  # Probability of adding random spikes to the data
timestamps = np.arange(n_points)  # Array of timestamps from 0 to n_points

# Streaming data generator
data_stream = stream_data(timestamps, trend_factor, seasonal_amplitude1, seasonal_amplitude2, frequency1, frequency2, noise_level, random_spike_prob)

# Initialize empty time series and lists for storing anomalies
time_series_list = []  # List to store the streamed time series data
first_layer_anomalies = []  # List to store anomalies detected in the first layer (Z-score and neighbor deviation)
confirmed_anomalies = []  # List to store anomalies confirmed by the second layer (Isolation Forest)

# Sliding window and anomaly detection parameters
sliding_window_size = 40  # Window size for calculating Z-score-based anomaly detection
neighbor_window_size = 10  # Window size for detecting deviations from neighboring points
z_threshold = 2  # Threshold for the Z-score-based anomaly detection
neighbor_threshold = 2 # Threshold for detecting neighbor deviations

# Enable interactive plotting mode
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

# Initialize plot lines for streamed data, first layer anomalies, and confirmed anomalies
streamed_line, = ax.plot([], [], label='Streamed Data', color='blue')  # Main data stream line
first_layer_scatter = ax.scatter([], [], color='orange', label='First Layer Anomalies')  # Scatter plot for first layer anomalies
confirmed_scatter = ax.scatter([], [], color='red', label='Confirmed Anomalies')  # Scatter plot for confirmed anomalies

ax.set_title('Real-Time Anomaly Detection with Isolation Forest')  # Set plot title
ax.legend(loc='best')  # Set legend location

# Process streaming data and update the plot in real time
for timestamp, data_point in data_stream:
    # Append the new data point to the time series list
    time_series_list.append((timestamp, data_point))
    # Create a pandas Series object from the time series list (timestamp as index, data point as value)
    time_series = pd.Series([dp for _, dp in time_series_list], index=[timestamp for timestamp, _ in time_series_list])

    # First layer anomaly detection: Z-score and neighbor deviation
    if len(time_series) >= sliding_window_size:
        # Z-score-based anomaly detection
        is_anomaly, z_score = detect_first_layer_anomalies(time_series, data_point, sliding_window_size, z_threshold)
        if is_anomaly:
            first_layer_anomalies.append((timestamp, data_point))  # Store detected anomaly
            print(f"Anomaly detected at {timestamp} with Z-Score: {z_score}")  # Print Z-score anomaly info

        # Neighbor deviation-based anomaly detection
        if detect_neighbor_deviation(time_series, data_point, neighbor_window_size, neighbor_threshold=neighbor_threshold):
            first_layer_anomalies.append((timestamp, data_point))  # Store detected anomaly
            print(f"Anomaly detected at {timestamp} by neighbor deviation.")  # Print neighbor deviation anomaly info

    # Second layer anomaly validation with Isolation Forest every 30 points
    if len(time_series) >= 50 and len(time_series) % 50 == 0:
        confirmed = validate_with_isolation_forest(time_series, first_layer_anomalies)  # Validate anomalies
        confirmed_anomalies.extend(confirmed)  # Add confirmed anomalies to the final list

    # Update plot lines with the new data points
    streamed_line.set_data(time_series.index, time_series.values)

    # Update first layer anomaly scatter plot if anomalies exist
    if first_layer_anomalies:
        first_layer_ts, first_layer_values = zip(*first_layer_anomalies)
        first_layer_scatter.set_offsets(np.c_[date2num(first_layer_ts), first_layer_values])

    # Update confirmed anomaly scatter plot if confirmed anomalies exist
    if confirmed_anomalies:
        confirmed_ts, confirmed_values = zip(*confirmed_anomalies)
        confirmed_scatter.set_offsets(np.c_[date2num(confirmed_ts), confirmed_values])

    # Adjust plot limits and refresh the plot
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

# Print all confirmed anomalies after the streaming is complete
print("\nAll Confirmed Anomalies:")
for timestamp, value in confirmed_anomalies:
    print(f"Timestamp: {timestamp}, Anomaly Value: {value}")

# Disable interactive plotting and show the final plot
plt.ioff()
plt.show()
