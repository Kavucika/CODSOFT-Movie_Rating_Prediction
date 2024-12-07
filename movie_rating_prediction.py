import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('E:/kavucika/codsoft/IMDb Movies India.csv/IMDb Movies India.csv', encoding='latin1')

print("Columns in dataset:", data.columns)
print("")
print("Initial data sample:")
print(data.head())
print("")
print("Missing values in each column:")
print(data.isnull().sum())

data = data.dropna(subset=['Rating'])

data['Duration'] = data['Duration'].str.replace(' min', '').astype(float)
data['Duration']=data['Duration'].fillna(data['Duration'].mean())

data['Votes'] = data['Votes'].str.replace(',', '').astype(float)
data['Votes'] = data['Votes'].fillna(data['Votes'].median())

data['Genre'].fillna('Unknown', inplace=True)
data['Director'].fillna('Unknown', inplace=True)
data['Actor 1'].fillna('Unknown', inplace=True)
data['Actor 2'].fillna('Unknown', inplace=True)
data['Actor 3'].fillna('Unknown', inplace=True)

print(data.isnull().sum())
print("")
print(data.head())

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

print("At encoded_features")
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

print("At numerical_features")
numerical_features = data[['Duration', 'Votes']]
X = pd.concat([numerical_features.reset_index(drop=True), encoded_features_df.reset_index(drop=True)], axis=1)
y = data['Rating']  

print("At training")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

print("At prediction")
y_pred = model.predict(X_test)

print("At calculating")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

user_input = {
    'Genre': input("Enter Genre: "),
    'Director': input("Enter Director: "),
    'Actor 1': input("Enter Actor 1: "),
    'Actor 2': input("Enter Actor 2: "),
    'Actor 3': input("Enter Actor 3: "),
    'Duration': float(input("Enter Duration (in minutes): ")),
    'Votes': float(input("Enter Votes: "))
}

user_df = pd.DataFrame([user_input])  
user_encoded = encoder.transform(user_df[categorical_features]) 
user_encoded_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(categorical_features))

user_features = pd.concat([user_df[['Duration', 'Votes']].reset_index(drop=True), user_encoded_df], axis=1)

predicted_rating = model.predict(user_features)
print(f"The predicted rating for the movie is: {predicted_rating[0]:.2f}")

