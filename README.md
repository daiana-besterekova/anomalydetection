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

## References
- https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html 
- https://towardsdatascience.com/isolation-forest-with-statistical-rules-4dd27dad2da9
- https://dataheroes.ai/glossary/z-score-for-anomaly-detection/#:~:text=Calculating%20standard%20deviation%3A%20The%20next,for%20everything%20within%20the%20dataset
- https://towardsdatascience.com/detecting-real-time-and-unsupervised-anomalies-in-streaming-data-a-starting-point-760a4bacbdf8

