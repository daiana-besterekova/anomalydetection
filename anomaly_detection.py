import numpy as np
from sklearn.ensemble import IsolationForest

# Z-score based first layer anomaly detection
def detect_first_layer_anomalies(time_series, data_point, sliding_window_size, z_threshold):
    """
    Detects anomalies in the data stream using Z-scores, which measure how far a point is from the mean 
    of a sliding window. A Z-score indicates how many standard deviations a data point is from the mean 
    of the recent window, and values above the given threshold are flagged as anomalies.

    Parameters:
    - time_series (pd.Series): The time series data up to the current point.
    - data_point (float): The current data point to be analyzed.
    - sliding_window_size (int): The number of previous data points used for calculating the mean and standard deviation.
    - z_threshold (float): The Z-score threshold. If the absolute value of the Z-score is greater than this threshold, 
                           the data point is flagged as an anomaly.

    Returns:
    - (bool, float): A tuple where the first value is True if the Z-score exceeds the threshold (indicating an anomaly), 
                     False otherwise. The second value is the Z-score of the current data point.
    """
    recent_window = time_series[-sliding_window_size:]
    rolling_mean = recent_window.mean()
    rolling_std = recent_window.std()
    
    # Calculate Z-score for the current data point
    if rolling_std > 0:
        z_score = (data_point - rolling_mean) / rolling_std
    else:
        z_score = 0

    # First layer anomaly detection
    if abs(z_score) > z_threshold:
        return True, z_score
    return False, z_score


# Neighbor deviation based anomaly detection
def detect_neighbor_deviation(time_series, data_point, neighbor_window_size, neighbor_threshold):
    """
    Detects anomalies by measuring how much the current data point deviates from the average 
    of its neighboring points. The deviation is calculated as the number of standard deviations 
    between the current point and the mean of the neighbor window. Points that deviate significantly 
    from the local neighbors are flagged as anomalies.

    Parameters:
    - time_series (pd.Series): The time series data up to the current point.
    - data_point (float): The current data point to be analyzed.
    - neighbor_window_size (int): The number of neighboring data points used for comparison.
    - neighbor_threshold (float): The threshold for deviation from the neighbor mean. If the deviation 
                                  exceeds this threshold, the point is considered an anomaly.

    Returns:
    - bool: True if the deviation from the neighbors exceeds the threshold, indicating an anomaly, False otherwise.
    """
    neighbors = time_series[-(neighbor_window_size + 1):-1]
    neighbor_mean = neighbors.mean()
    neighbor_std = neighbors.std()
    
    if neighbor_std > 0:
        deviation_from_neighbors = abs(data_point - neighbor_mean) / neighbor_std
    else:
        deviation_from_neighbors = 0

    # Anomaly detection based on neighbor deviation
    return deviation_from_neighbors > neighbor_threshold

# Second layer: Isolation Forest based anomaly validation
def validate_with_isolation_forest(time_series, first_layer_anomalies, window_size=20):
    """
    Validates the anomalies detected in the first layer using the Isolation Forest algorithm. Isolation 
    Forest identifies anomalies by isolating data points that are different from the rest of the data. It 
    operates by randomly partitioning the data, and points that require fewer partitions to isolate are 
    flagged as anomalies. This method helps confirm whether first-layer anomalies are genuine.

    Parameters:
    - time_series (pd.Series): The time series data up to the current point.
    - first_layer_anomalies (list): A list of tuples containing timestamps and values for anomalies detected 
                                    in the first layer.
    - window_size (int): The size of the rolling window used to calculate residuals for Isolation Forest training.

    Returns:
    - list: A list of confirmed anomalies after Isolation Forest validation, each represented as a tuple 
            (timestamp, value). If no anomalies are confirmed, returns an empty list.
    """
    # Train Isolation Forest on residuals
    residuals = time_series - time_series.rolling(window=window_size).mean()
    residuals_clean = residuals.dropna().values.reshape(-1, 1)
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_forest.fit(residuals_clean)

    # Validate first layer anomalies
    if first_layer_anomalies:
        anomalies_ts, anomalies_values = zip(*first_layer_anomalies)
        residuals_anomalies = np.array(anomalies_values).reshape(-1, 1)
        is_anomaly = isolation_forest.predict(residuals_anomalies)
        confirmed_anomalies = [(anomalies_ts[i], anomalies_values[i]) for i, val in enumerate(is_anomaly) if val == -1]
        return confirmed_anomalies
    return []