
import pandas as pd
from sklearn.model_selection import train_test_split

'Initalize Data'
df = pd.read_csv("heart_statlog.csv")

x_axis = df[["age", "sex", "chest pain type", "resting bp s",
        "cholesterol", "fasting blood sugar", "resting ecg",
        "max heart rate", "exercise angina", "oldpeak", "ST slope"]]

y_axis = df["target"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test, = train_test_split(x_axis, y_axis, test_size=0.2, random_state=42)