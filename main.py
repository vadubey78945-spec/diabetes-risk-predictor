import pickle
from src.data_preprocessing import load_and_clean_data
from src.train_model import train_model
from src.evaluate import evaluate_model


#Load and clean the data
df = load_and_clean_data('data/pima_diabetes_data.csv')

#Train the model
model, scaler, X_test, y_test =train_model(df)

#Evaluste the model
evaluate_model(model, scaler, X_test, y_test)

#Save the model and the scaler for furure use
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Model and scaler saved successfully.")