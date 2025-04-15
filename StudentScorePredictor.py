import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pickle



df = pd.read_csv("student_data.csv")
#print(df.head())
#print(df.describe())LabELB-2074108018.us-east-1.elb.amazonaws.com

# Drop irrelevant text columns
#df = df.drop(["age","Walc", "freetime", "famrel", "guardian", "internet"], axis=1)

# Separate target
#y = df["G3"]
#X = df.drop("G3", axis=1)
selected_features = ['G1', 'G2', 'failures', 'studytime', 'absences']
X = df[selected_features]
y = df['G3']

# Convert categorical columns using one-hot encoding
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Predict and evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
#print(df.corr(numeric_only=True)["G3"].sort_values(ascending=False))


 #Save the trained model to a file
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)
