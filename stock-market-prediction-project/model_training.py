from sklearn.ensemble import RandomForestClassifier
import pandas as pd

print("model_training.py is being imported")

def train_model(data, predictors):
    print("train_model function is called")
    try:
        # Create a Random Forest model
        model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
        
        # Split the data into training and testing sets
        train = data.iloc[:-100]
        test = data.iloc[-100:]
        
        # Train the model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions on the test set
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        
        return model, preds, test
    except Exception as e:
        print("An error occurred in train_model:", e)
        raise

def evaluate_model(test, preds):
    print("evaluate_model function is called")
    from sklearn.metrics import precision_score
    
    precision = precision_score(test["Target"], preds)
    print(f"Model Precision: {precision:.2f}")
    return precision

if __name__ == "__main__":
    print("model_training.py is being run directly")