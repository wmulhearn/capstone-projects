import pandas as pd

def backtest(data, model, predictors, start=1000, step=750):
    """
    Backtest the model to make predictions across the dataset.
    """
    predictions = []
    
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit the model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions using predict_proba
        preds = model.predict_proba(test[predictors])[:, 1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > 0.6] = 1
        preds[preds <= 0.6] = 0
        
        # Combine predictions and actual values
        combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
        predictions.append(combined)
    
    return pd.concat(predictions)

def evaluate_backtest(predictions):
    """
    Evaluate the backtested predictions using precision.
    """
    from sklearn.metrics import precision_score
    
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Backtest Precision: {precision:.2f}")
    return precision