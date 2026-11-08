'Model: Logistic Regression Feature Importance'

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

