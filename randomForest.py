from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

with open('labelsTraining.txt', 'r') as file:
   labels = pd.read_csv('labelsTraining.txt', sep=';')

with open('featureMatrixTraining.txt', 'r') as file:
   featureMatrix = pd.read_csv('featureMatrixTraining.txt', sep=',')

y = labels[:].values.ravel()

X = featureMatrix.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

random_forest = RandomForestClassifier(n_estimators=500, max_features=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
