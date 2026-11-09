'Model: Logistic Regression Feature Importance'

"""
This model observes the relationship between individual health markers and correlation to having heart disease.

Logistic Regression assigns a coefficient to each feasture, indicating how that feature infleunce's the model's prediction.

Large coefficients indicate stronger influence on model's predictions. 
The strongest features are ST slope (aprox. 1.2), sex (aprox. 1.2), and angina (.aprox 1.0),
while chloesterol (virtually none), resting bp (virtually none) , and heartrate (almost none) have the weakest correlation.

Logistic Regression helps understandwhat model learned from training splits.

The coefficients are organized into a dataframe and are sorted from highest to lowest absolute importance to help better observe how features compare
in correlation to target (heart disease).
"""
from heart_data import df, x_axis, x_train, y_train
from sklearn.linear_model import LogisticRegression

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

importance = pd.DataFrame({
    'Feature': x_axis.columns,
    'Coefficient': model.coef_[0],
    'Absolute Importance': abs(model.coef_[0])
})

importance = importance.sort_values(
    by='Absolute Importance',
    ascending=False
)


print(importance)

plt.barh(importance['Feature'], importance['Absolute Importance'])
plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")
plt.title("Feature Importance - Logistic Regression")
plt.gca().invert_yaxis()
plt.show()

