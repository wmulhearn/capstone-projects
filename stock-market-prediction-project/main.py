import data_processing
import model_training
import backtesting

# Step 1: Download and preprocess the data
stock_data = data_processing.download_data()
data = data_processing.preprocess_data(stock_data)

# Define the predictors (features)
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Step 2: Train the model and evaluate it on the last 100 rows
model, preds, test = model_training.train_model(data, predictors)
model_training.evaluate_model(test, preds)

# Step 3: Backtest the model over the entire dataset
predictions = backtesting.backtest(data, model, predictors)
backtesting.evaluate_backtest(predictions)