import pandas as pd
from sklearn.linear_model import LinearRegression

# House price dataset
data = {
    'Square_feet': [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5],
    'Bathrooms': [1, 2, 2, 3, 3, 4, 4],
    'Price': [150000, 180000, 220000, 270000, 300000, 340000, 400000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Square_feet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
sqft = float(input("Enter Square Feet: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))

# Create DataFrame for prediction
new_house = pd.DataFrame({
    'Square_feet': [sqft],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms]
})

# Predict price
predicted_price = model.predict(new_house)

# Display result
print("Predicted House Price: ₹", round(predicted_price[0], 2))