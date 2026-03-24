from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    X = df.drop("Outcome", axis =1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test