import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("students.csv")

# Selecting features and target variable
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "student_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model training complete. Saved as 'student_model.pkl'")



