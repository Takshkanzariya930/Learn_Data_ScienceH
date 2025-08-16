import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_FILE = "wine_model.pkl"
PIPELINE_FILE = "wine_pipeline.pkl"

def build_pipeline(data):
    
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    return pipeline.fit_transform(data)

if (not os.path.exists(MODEL_FILE)) or (not os.path.exists(PIPELINE_FILE)):
    
    data = pd.read_csv("WineQuality/wine_quality_white.csv")
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    strat_train_set = pd.DataFrame([0])
    strat_test_set = pd.DataFrame([0])
    
    for train_index, test_index in split.split(data, data["quality"]):
        strat_train_set = data.loc[train_index].drop("quality", axis=1)
        strat_test_set = data.loc[test_index]
        
    training_data = strat_train_set.copy()
    
    pipeline = build_pipeline(training_data)
    
    model = 