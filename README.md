# Forecasting Bitcoin Prices with Time Series Analysis

## Introduction
Time series forecasting is an essential tool in financial markets, especially for volatile assets like Bitcoin (BTC). Accurate BTC price predictions provide valuable insights for investors, traders, and analysts, allowing for data-driven decisions and enhanced risk management strategies. Leveraging advanced time series forecasting models, such as Long Short-Term Memory (LSTM) networks, helps capture complex temporal dependencies in Bitcoin’s price history, making this approach highly effective in identifying and anticipating market trends.

---

## 1. An Introduction to Time Series Forecasting
Time series forecasting involves using historical data to predict future values based on observed trends and patterns. In the context of financial markets, accurate forecasting is vital for anticipating price fluctuations, market dynamics, and volatility. Given Bitcoin’s unique characteristics, forecasting BTC prices can provide significant value by informing investment strategies, portfolio management, and market analysis. This approach offers a systematic way to capture trends in an otherwise unpredictable asset class, enabling users to leverage historical data to make informed predictions.

---

## 2. Preprocessing Method
Data preprocessing is critical for improving the model’s ability to learn and predict. The following preprocessing steps were used to prepare the BTC dataset:

- **Data Splitting**: A sequential split was performed (80/20) for training and testing to maintain temporal order, which is necessary as future predictions rely on past data.
- **Target Feature Extraction**: We selected the closing price as the primary feature for prediction since it reflects the market’s sentiment at the end of each day. The feature was renamed "price" for coding convenience.
- **Normalization (MinMaxScaler)**: The data was scaled between 0 and 1 using `MinMaxScaler`, enhancing model convergence by standardizing price values—a critical step for LSTM models.
- **Sequence Length (60 Timesteps)**: A 60-day window was chosen to capture short- to medium-term trends, allowing the model to learn patterns effectively.
- **Prediction Target (Next-Day Price)**: The model was designed to forecast the closing price for the next day, aligning with short-term forecasting needs in finance.

### Code Example
```python
from sklearn.preprocessing import MinMaxScaler

# Normalize the closing price
scaler = MinMaxScaler(feature_range=(0, 1))
data['price'] = scaler.fit_transform(data[['closing_price']])

# Prepare sequences with 60-day windows
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(data)):
    X.append(data['price'][i-sequence_length:i].values)
    y.append(data['price'][i])
```

3. Setting Up tf.data.Dataset for Model Inputs

I used TensorFlow’s tf.data.Dataset API to efficiently structure the dataset, which facilitated data handling and processing for the LSTM model.
	•	Windowing: Each sample was divided into a 60-day window to provide the model with context on sequential patterns within that timeframe.
	•	Batching: With a batch size of 32, we optimized memory usage while speeding up the training process.
	•	Shuffling: While the order of timesteps remained intact to maintain temporal sequence, batches were shuffled to prevent the model from overfitting to specific sequences, improving generalization.

Code Example
```
import tensorflow as tf

# Create tf.data.Dataset from sequences
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Windowing and batching
sequence_length = 60
dataset = dataset.window(sequence_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(sequence_length))
dataset = dataset.batch(32).shuffle(1000)
```
4. Model Architecture

I choose an LSTM-based model for BTC price forecasting due to its capacity to capture long-term dependencies in sequential data. LSTMs are ideal for time series analysis as they retain information over long sequences through their internal memory gates, which is crucial for financial forecasting.
	•	First LSTM Layer (50 units): Outputs a sequence, enabling further layers to process more granular temporal patterns.
	•	Second LSTM Layer (50 units): Compresses the sequence into a single timestep output, capturing essential patterns in a compact representation.
	•	Dense Layer: A fully connected layer with 25 units to refine the features extracted from the LSTM layers.
	•	Output Layer: A final dense layer with a single unit to predict the next day’s BTC price.

Code Example

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

5. Results and Evaluation

I evaluated the model’s performance using metrics and visualizations to compare predicted vs. actual BTC prices.
	•	Performance Metrics:
	•	Root Mean Squared Error (RMSE): 3309.22, showing a relatively low error magnitude.
	•	Mean Absolute Error (MAE): 1777.60, indicating minimal average deviation from actual values.
	•	R-squared (R²) Score: 0.965, implying that the model captures 96.5% of the variance in BTC prices.
	•	Predicted vs Actual Values Visualization:
The plot of predicted vs. actual values shows that the model accurately follows BTC price trends, with predicted values (in red) closely tracking actual prices (in blue). This alignment indicates effective learning of BTC price patterns.
	•	Residual Analysis:
The residual plot and histogram show stable and randomly distributed errors, suggesting the model’s reliability. The near-normal distribution of residuals reflects that errors are unbiased, enhancing confidence in the model’s predictive capability.

Visualization Example

```
import matplotlib.pyplot as plt

plt.plot(actual_prices, label='Actual')
plt.plot(predicted_prices, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
```

![Model Loss](https://github.com/user-attachments/assets/807b99fa-7601-476a-8b6c-10c0d2310ff9)


6. Conclusion

The LSTM model for BTC price forecasting demonstrates high accuracy and robustness across varying market conditions. By capturing complex temporal patterns, the model offers valuable insights into BTC price trends and aids in short-term predictions. However, due to Bitcoin’s inherent volatility, ongoing monitoring and periodic retraining will be essential for maintaining performance. The model could be further enhanced by incorporating additional features, such as trading volume or sentiment analysis, to better adapt to sudden market shifts. Overall, this LSTM model provides a solid foundation for BTC price forecasting, establishing a reliable tool for understanding and anticipating Bitcoin market dynamics.

