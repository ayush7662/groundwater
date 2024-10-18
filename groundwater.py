import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('groundwater.csv')

# Step 1: Data Preprocessing
# Handle missing values by filling them with the median
df.fillna(df.median(), inplace=True)

# Encode 'WellIdentifier' (categorical feature) using LabelEncoder
encoder = LabelEncoder()
df['WellIdentifier'] = encoder.fit_transform(df['WellIdentifier'])

# Separate features (X) and target (y)
X = df.drop(columns=['AnnualAverageWaterLevel'])
y = df['AnnualAverageWaterLevel']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Building
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Display model coefficients
print("Model Coefficients:", model.coef_)
