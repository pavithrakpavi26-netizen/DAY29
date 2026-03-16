import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load dataset (keep file in same folder)
titanic_data = pd.read_csv('titanic.csv')

# Remove missing survival values
titanic_data = titanic_data.dropna(subset=['Survived'])

# Select features
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = titanic_data['Survived']

# Convert categorical to numeric
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# Fill missing age values
X['Age'] = X['Age'].fillna(X['Age'].median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Prediction
y_pred = rf_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)

# Test with one sample
sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print("\nSample Passenger:", sample_dict)

if prediction[0] == 1:
    print("Predicted Survival: Survived")
else:
    print("Predicted Survival: Did Not Survive")