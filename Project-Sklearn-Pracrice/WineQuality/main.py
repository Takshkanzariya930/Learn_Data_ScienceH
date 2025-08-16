import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "WineQuality/wine_model.pkl"
PIPELINE_FILE = "WineQuality/wine_pipeline.pkl"

def build_pipeline():
    
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return pipeline

if (not os.path.exists(MODEL_FILE)) or (not os.path.exists(PIPELINE_FILE)):
    
    data = pd.read_csv("WineQuality/wine_quality_white.csv")
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    strat_train_set = pd.DataFrame([0])
    strat_test_set = pd.DataFrame([0])
    
    for train_index, test_index in split.split(data, data["quality"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        
    training_data = strat_train_set.drop("quality", axis=1).copy()
    data_labels = strat_train_set["quality"]
    
    pipeline = build_pipeline()
    prepared_data = pipeline.fit_transform(training_data)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(prepared_data, data_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("Model trained and saved")
    
    
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

input_data = pd.read_csv("WineQuality/strat_test_set.csv").drop("quality", axis=1)
transformed_data = pipeline.transform(input_data)
predictions = model.predict(transformed_data)

input_data["quality"] = predictions
    
input_data.to_csv("WineQuality/output.csv", index=False)