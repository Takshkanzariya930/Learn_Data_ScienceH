import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

MODEL_FILE = "BanknoteAuth/note_model.pkl"
PIPELINE_FILE = "BanknoteAuth/note_pipeline.pkl"

def build_pipeline():
    return Pipeline([("scaler", StandardScaler())])

if (not os.path.exists(MODEL_FILE)) or (not os.path.exists(PIPELINE_FILE)):
    
    data = pd.read_csv("BanknoteAuth/data_banknote_authentication.csv")
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    training_data = pd.DataFrame([0])
    
    for train_index, test_index in split.split(data, data["Y"]):
        training_data = data.loc[train_index]
        
    data_labels = training_data["Y"]
    training_data = training_data.drop("Y", axis=1)
    
    pipeline = build_pipeline()
    training_data = pipeline.fit_transform(training_data)
    
    model = RandomForestClassifier()
    model = model.fit(training_data, data_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("Model trained and saved")

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

input_data = pd.read_csv("BanknoteAuth/strat_test_set.csv").drop("Y", axis=1)

transformed_data = pipeline.transform(input_data)

predictions = model.predict(transformed_data)

input_data["pY"] = predictions

input_data.to_csv("BanknoteAuth/output.csv", index=False)