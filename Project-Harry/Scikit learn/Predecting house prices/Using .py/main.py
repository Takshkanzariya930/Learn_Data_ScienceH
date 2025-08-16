import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(cat_attributes, num_attributes):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    strat_train_set = pd.DataFrame([0])
    
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        
    housing = strat_train_set.copy()

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1).copy()
    
    num_attributes = housing_features.drop("ocean_proximity", axis=1).columns.to_list()
    cat_attributes = ["ocean_proximity"]
    
    pipeline = build_pipeline(cat_attributes,num_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("Model trained and saved")
    
else:
    
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("strat_test_set.csv").drop("median_house_value", axis=1)
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions
    
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")