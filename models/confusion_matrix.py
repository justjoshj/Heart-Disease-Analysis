'Model: Logistic Regression Confusion Matrix '

"""
This model graphs the performance of trainined logistic regression model by cxomparing to its predicted classifications against actual target value sfrom tes dataset

The confisuoon matrix organizes predicitons int oforu categories: true postiives, true negative, false postives, false negatives. 

True positives ad true negatives indicate correct predictions by model, while false ones indicate incorrect predictions. 

True positve scored: 90, True negative: 115, False positive: 16, False negative: 17

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.84      0.85       107
           1       0.87      0.88      0.87       131

    accuracy                           0.86       238
   macro avg       0.86      0.86      0.86       238
weighted avg       0.86      0.86      0.86       238

From the classification report, we can conclude that our legistic model is performing overall well.

Accuracy 86% - The model correctly classified 86 percent of all 238 patients
Precision 87% - of the patients predicted to have heart disease, 87% actually had it.
F1 score - balance of precition and recall
Recall - Model correctly idenfitied 88% of paitents who actually had heart disease

We can conclude our model had relatively consisnte performance, detection, and limited classification errors. 
"""
from heart_data import x_train, x_test, y_train, y_test

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=['No', 'Yes'],
    cmap='Blues'
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()