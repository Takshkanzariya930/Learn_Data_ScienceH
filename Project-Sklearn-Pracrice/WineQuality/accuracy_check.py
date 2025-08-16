import pandas as pd

test_set = pd.read_csv("WineQuality/strat_test_set.csv")
output_set = pd.read_csv("WineQuality/output.csv")

actual_values = test_set["quality"].values.tolist()
predicted_values = output_set["quality"].values.tolist()

count = 0
for i in range(len(actual_values)):
    
    if actual_values[i] == predicted_values[i]:
        count = count + 1

print((count/len(actual_values))*100)