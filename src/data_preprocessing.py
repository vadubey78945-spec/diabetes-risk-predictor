import pandas as pd

def load_and_clean_data(data_file):
    df = pd.read_csv(data_file)

    #Columns where 0 is invalid
    cols_with_zero = ['DiabetesPedigreeFunction','Pregnancies', 'Glucose', 
                      'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    #for col in cols_with_zero:
    #     print(f"{col} has {(df[col] == 0).sum()} zero values")

    for col in cols_with_zero:
         median = df[col].median()
         df[col] = df[col].replace(0, median)

    return df