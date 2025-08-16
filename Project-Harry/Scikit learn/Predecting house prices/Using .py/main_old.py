import pandas as pd
import numpy as np
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

# 1. Load the Dataset
housing = pd.read_csv("../housing.csv")

# 2. Creating Stratified test set 
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

strat_train_set = pd.DataFrame([0])
strat_test_set = pd.DataFrame([0])

for train_index, test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
    
housing = strat_train_set.copy()

# 3. Separate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1).copy()

# 4. List the numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# 5. Making Pipeline 
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# 7. Train the model 

# Linear regression

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(housing_prepared, housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_pred)
lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(f"Linear Regression RMSE:")
print(pd.Series(lin_rmse).describe())


# Decision Tree

dec_reg = DecisionTreeRegressor()
dec_reg = dec_reg.fit(housing_prepared, housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_pred)
dec_rmse = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print("Decision Tree RMSE:")
print(pd.Series(dec_rmse).describe())


# Random forest regressor

random_forest_reg = RandomForestRegressor()
random_forest_reg = random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_pred = random_forest_reg.predict(housing_prepared)
random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_pred)
print(random_forest_rmse)
random_forest_rmse = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print("Random Forest RMSE:")
print(pd.Series(random_forest_rmse).describe())
