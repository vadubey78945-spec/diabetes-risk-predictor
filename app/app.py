import streamlit as st 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", category= UserWarning)


# load the trained model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler =pickle.load(open("scaler.pkl", "rb"))

st.markdown("##")

# Load dataset for Benchmarking
df = pd.read_csv("data/pima_diabetes_data.csv")

# Setting up the Streamlit app configuration
st.set_page_config(page_title = "Diabetes Risk Predictor", 
                   page_icon = ":drop_of_blood:", 
                   layout = "wide")

#Glassmorphism background
st.markdown("""
            <style>
            .stApp{
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color:white;
            }
            
            .card{
            background: rgba(255,255,255,0.050);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            }
            
            h1, h2, h3{
            color: #00C6FF;
            }

            /* GENERAL TEXT FIX */
            label, .stNumberInput label, .stTextInput label {
            font-weight: 600;
            color: white !important;
            }

            /* PRIMARY BUTTON STYLE */
            
            .stButton button {
            background: linear-gradient(90deg, #4F46E5,#4338CA) !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            border: none;
            box-shadow: 0px 0px 15px rgba(79, 70, 229, 0.3);
            transition: all 0.3s ease;
            }

            /* HOVER EFFECT */
            .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 25px rgba(67, 56, 202, 0.5);
            transform: translateY(-5px);
            }

            /* SECONDARY BUTTON (Download) */
            .stDownloadButton button {
            background: linear-gradient(90deg, #00ff95, #00c853) !important;
            color: black !important;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            border: none;
            }

            /* HOVER */
            .stDownloadButton button:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 15px rgba(0,255,150,0.5);
            }
            </style>
            """,unsafe_allow_html=True)


# Title and Description
st.title("Diabetes Risk Predictor")
st.markdown("### Advanced Diabetes Risk Analysis")

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class= "card">', unsafe_allow_html=True)
    st.subheader("Patient Inputs")

    col1, col2, = st.columns(2)

    with col1:
        preg = st.number_input("Number of Pregnancies", min_value = 0, max_value = 20, value = 0)
        glucose = st.number_input("Glucose level", min_value = 0, max_value = 200, value = 0)
        BP = st.number_input("Blood Pressure", min_value = 0, max_value = 200, value = 0)
        skin = st.number_input("Skin Thickness", min_value = 0, max_value = 100, value = 0)


    with col2:
        insulin = st.number_input("Insulin level", min_value = 0, max_value = 900, value = 0)
        BMI = st.number_input("BMI", min_value = 0.0, max_value = 70.0, value = 0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value = 0.0, max_value = 3.0, value = 0.0)
        age = st.number_input("Age", min_value = 0, max_value =120, value = 0)      


    
    analyze = st.button("Analyze Risk")
    st.markdown('</div>', unsafe_allow_html=True)

#Gauge Function
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode = "gauge + number",
        value = prob * 100,
        title = {'text': "Diabetes Risk (%)"},
        gauge = {
            'axis':{'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.2},
            'steps': [
                {'range': [0,30], 'color': "green"},
                {'range': [30,70], 'color': "orange"},
                {'range': [70,100], 'color': "red"}
            ],
            'threshold':{
                'line': {'color': "white", 'width': 4},
                'value': prob * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=300
    )
    return fig

#Main Prediction Logic

with right:
   st.subheader("Risk Analysis")
   if analyze:

    # Create input dataframe for prediction 
    input_df = pd.DataFrame([[preg, glucose, BP, skin, insulin, BMI, dpf, age]],
                            columns = ["Pregnancies", "Glucose", "BloodPressure", 
                                       "SkinThickness", "Insulin", "BMI", 
                                       "DiabetesPedigreeFunction", "Age"])

    #Scale the input data
    input_scaled = scaler.transform(input_df)

    #Get predicted probability for the positive class (diabetes)
    prob = model.predict_proba(input_scaled)[0][1]

    #Get binary prediction based on default threshold of 0.5
    pred = model.predict(input_scaled)[0]

    st.plotly_chart(create_gauge(prob), width = "stretch")

    #Risk Level
    if prob < 0.3:
        st.success("Low Risk")

    elif prob < 0.7 :
        st.warning("Medium Risk")

    else:
        st.error("High Risk")

        if pred==1:
            st.error("High risk of diabetes. Please consult a healthcare professional for further evaluation.")
        else:
            st.success("Low risk of diabetes. However, it is always a good idea to maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.")

    #Download Report
    st.markdown("---")
    st.subheader("Download Report")
    report = f"""
    Diabetes Risk Report
    Risk: {prob*100:.2f}%
    Glucose:{glucose}
    BMI:{BMI}
    Age:{age}
    """

    st.download_button(
        label="Download Report",
        data=report,
        file_name ="health_report.txt"
    )

#Feature Importance
st.markdown("---")
importance =model.feature_importances_
features = ["Pregnancies", "Glucose", "BloodPressure", 
            "SkinThickness", "Insulin", "BMI", 
            "DiabetesPedigreeFunction", "Age"]

#Sort features by importance for better visualization
sorted_idx = np.argsort(importance)
sorted_importance = importance[sorted_idx]
sorted_features = np.array(features)[sorted_idx]

#Plot the feature Importance using a horizontal bar chart
fig, ax = plt.subplots()
ax.barh(sorted_features, sorted_importance, color = "skyblue")

#Add labels and title
ax.set_xlabel("Importance")
ax.set_title("Feature Importance", color ="Blue", fontsize = 23)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(features)[sorted_idx])
st.pyplot(fig)

st.markdown("##")

#Benchmarking Section
st.markdown("---")
st.subheader("Health Benchmarking")

avg_glucose = df["Glucose"].mean()
avg_bmi = df["BMI"].mean()

compare_df = pd.DataFrame({
    "Metric": ["Glucose", "BMI"],
    "You": [glucose, BMI],
    "Average": [avg_glucose, avg_bmi]
}) 

st.bar_chart(compare_df.set_index("Metric"))

st.markdown("##")

#Explanation 
st.subheader("Why this prediction?")
st.write("""The model considers glucose, BMI, and age as key factors.
         Higher glucose and BMI significanty increase diabetes risk.
         Age also contributes but to a lesser extent.""")

st.markdown("##")

#Smart Insights
st.markdown("---")
st.subheader("Health Insights and Recommendations")

if glucose > avg_glucose:
    st.warning("Your glucose level is above average. Consider regular monitoring and a balanced diet.")
if BMI > avg_bmi:
    st.warning("Your BMI is above average. Consider regular exercise and a healthy diet to manage your weight.")
if glucose < avg_glucose and BMI < avg_bmi:
    st.success("Your health metrics are within a good range. Keep up the healthy lifestyle!")

st.markdown("##")


#Footer
st.markdown("---" *100)
st.caption("Health Analysis System developed by Vaibhav Dubey. For educational purpose only. Not a substitue for professional medical advice. Always consult a healthcare provider for medical concerns.")
st.caption("Not a medical diagnosis tool.")