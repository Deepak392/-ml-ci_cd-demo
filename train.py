print("ML pipeline running in GitHub Actions")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)

print("Model trained successfully")
print("Accuracy:", model.score(X, y))
