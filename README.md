# Efficient Data Stream Anomaly Detection

## Steps to Run the Project
1. Clone the Repository If your project is hosted on GitHub or another version control platform, clone it by running:
   `git clone https://github.com/daiana-besterekova/anomalydetection.git`
2. Then navigate into the project folder `cd project_name` (change `project_name` to how it is saved on your computer)
3. Set Up a Virtual Environment (Optional but Recommended) Create a virtual environment to keep the dependencies isolated:
`python -m venv venv`
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows:  `.\venv\Scripts\activate` 
4. Install Dependencies Install all the required packages specified in requirements.txt: `pip install -r requirements.txt`
5. Run the Project Run the main script to start your data stream and anomaly detection processes: `python main.py`
6. Deactivate the Virtual Environment (Optional): `deactivate`


## Algorithm Explanation
In this project, I used a two-layer approach for anomaly detection, combining Z-score validation with neighbor threshold and Isolation Forest to enhance the accuracy of detecting anomalies in real-time data streams.

**1. Z-Score and Neighbor Threshold Validation**

Initially, I implemented a Z-score based method to detect anomalies. The Z-score measures how far a data point deviates from the mean, providing a simple way to spot unusual values in the stream. To improve the reliability of this approach, I included a neighbor threshold check. By analyzing adjacent data points, the model can better assess if an anomaly is contextually valid or simply an outlier due to short-term noise.

This combination effectively filters out false positives by ensuring anomalies are validated within the context of their surrounding data points, rather than isolated spikes.

**2. Isolation Forest for Second-Layer Validation**

The second layer of validation involves Isolation Forest, a robust machine learning algorithm specialized in identifying outliers. For every data point classified as an anomaly by the Z-score method, Isolation Forest further validates whether the point is truly anomalous based on how easy it is to "isolate" within the feature space.

Using this two-layer system significantly reduces the rate of false positives, as only anomalies detected by both methods are flagged for review.

**3. Challenges with STL Decomposition and Real-Time Streaming**

While working on this project, I initially explored using STL (Seasonal-Trend Decomposition) to separate out seasonal components from the data. However, STL requires the entire dataset to perform decompositions, making it unsuitable for our real-time streaming scenario. Since I needed a method that could process data as it arrives, I opted for the simpler Z-score approach combined with Isolation Forest, which works efficiently on real-time streams without requiring complete datasets upfront.

This combination of methods ensures effective and efficient anomaly detection while accommodating the real-time nature of the data.

## Data Stream Pattern and Visualization
The data stream in this project was generated experimentally using a sinusoidal function to imitate Requirement 2, which specifies "incorporating regular patterns, seasonal elements, and random noise." By adjusting the frequency and amplitude of the sine wave, I was able to create a base pattern that simulates the regularity of natural processes. On top of this, I introduced random noise and spikes to mimic real-world data with occasional anomalies.

**Visualizations and Anomaly Detection**

In the two visualizations produced by the project, we observe the detection and classification of anomalies:

- Confirmed anomalies are shown in red, representing data points flagged by both the Z-score/neighbor threshold layer and the second layer of Isolation Forest.
- Unconfirmed anomalies (those detected by deviations in the first layer but not validated by Isolation Forest) are marked in orange. This design choice allows for human validation, offering the flexibility to investigate points that are flagged initially but not confirmed.

While some points in the dataset may appear mislabeled as anomalies, this is an expected outcome. Given that the algorithm is unsupervised, it does not have prior knowledge of the correct classifications and learns over time as the simulation progresses.


![Figure 1](https://github.com/daiana-besterekova/anomalydetection/blob/main/Figure_1.png)

![Figure 2](https://github.com/daiana-besterekova/anomalydetection/blob/main/Figure_2.png)

**Adjustability of Parameters**

The parameters that define the patterns in the data stream, such as the frequency of the sinusoidal wave, the amount of noise, and the thresholds for detecting anomalies, can be easily changed and adjusted as needed. This flexibility allows the system to be fine-tuned to various real-world scenarios and datasets, making it adaptable for different use cases.

This method ensures that the model remains scalable while maintaining the ability to process real-time data streams efficiently, even with the challenges of noise and seasonal variation.

## References
- https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html 
- https://towardsdatascience.com/isolation-forest-with-statistical-rules-4dd27dad2da9
- https://dataheroes.ai/glossary/z-score-for-anomaly-detection/#:~:text=Calculating%20standard%20deviation%3A%20The%20next,for%20everything%20within%20the%20dataset
- https://towardsdatascience.com/detecting-real-time-and-unsupervised-anomalies-in-streaming-data-a-starting-point-760a4bacbdf8
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/
