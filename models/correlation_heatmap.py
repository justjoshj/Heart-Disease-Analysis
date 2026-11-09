'Model: Correlation Heatmap'

"""
Correlation heatmap observes relationship between different health markers in dataset, including relationship to heart disease.

A correlation coefficient is calculated for each pir of variables ranging from -1.0 to +1.0. Positive values indicate two variablers tend to increase together (postive association)
while negative values indicate that one variables tends to decrease as other increases (negative association). 

The heatmap uses color intensity to showcase each relationship to help compare. This model highkits what markers are strongest tied to heart diease, and those least tied.

Correlation of course, does not imply causation. (Unlike feature importance) It does not necessarily indicate predictive importance.

ST slope, exercsie angina, chest pain type, oldpeak, sex, age, and blood sugar (Fasted) all have positive correlation with heart disease.
Features like cholesterol and max herat rate had negative relationship.

ST slope had strongest postive correlation (+0.51) while max heart rate had strongest negative (-0.41)
"""
from heart_data import df

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Heart Disease Feature Correlation Heatmap")
plt.show()

