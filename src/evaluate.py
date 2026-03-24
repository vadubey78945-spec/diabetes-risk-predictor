from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, scaler, X_test, y_test):
    #Get predicted probablilities for the positive class
    y_prob= model.predict_proba(X_test)[:, 1]

    #Apply a custom threshold of 0.3 to convert probabilities to binary probabilities
    y_pred_new = (y_prob >0.3).astype(int)
    
    #Evaluate the model with new threshold 

    
    print("Confusion MAtrix:")
    print(confusion_matrix(y_test, y_pred_new))
    print("\n" +"=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_new))