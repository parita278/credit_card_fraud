import pandas as pd 
import pickle
import streamlit as st

st.set_page_config(
    page_title="Credit card Fraud Predictor",
    layout="centered"
)

# Load the trained model
with open("Model/creditcard_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# App title and description
st.title("Credit Card Fraud Prediction App")
st.markdown("Enter the credit Card details below to get a Fraud prediction.")

if model is not None:
    st.markdown("<h2 style='text-align: center;'>📊 Credit Card Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.text_input("Enter Time", placeholder="e.g. 0.0, 17237.0", key='time')
        v1 = st.text_input("Enter V1", placeholder="e.g. 1.34563,-11.648274", key='v1')
        v2 = st.text_input("Enter V2", placeholder="e.g. 1.34563,-11.648274", key='v2')
        v3 = st.text_input("Enter V3", placeholder="e.g. 1.34563,-11.648274", key='v3')
        v4 = st.text_input("Enter V4", placeholder="e.g. 1.34563,-11.648274", key='v4')
        v5 = st.text_input("Enter V5", placeholder="e.g. 1.34563,-11.648274", key='v5')
        v6 = st.text_input("Enter V6", placeholder="e.g. 1.34563,-11.648274", key='v6')
        v7 = st.text_input("Enter V7", placeholder="e.g. 1.34563,-11.648274", key='v7')
        v8 = st.text_input("Enter V8", placeholder="e.g. 1.34563,-11.648274", key='v8')
        v9 = st.text_input("Enter V9", placeholder="e.g. 1.34563,-11.648274", key='v9')
        v10 = st.text_input("Enter V10", placeholder="e.g. 1.34563,-11.648274", key='v10')
        v11 = st.text_input("Enter V11", placeholder="e.g. 1.34563,-11.648274", key='v11')
        v12 = st.text_input("Enter V12", placeholder="e.g. 1.34563,-11.648274", key='v12')
        v13 = st.text_input("Enter V13", placeholder="e.g. 1.34563,-11.648274", key='v13')
        v14 = st.text_input("Enter V14", placeholder="e.g. 1.34563,-11.648274", key='v14')
        
    
    with col2:
        amount = st.text_input("Enter Amount", placeholder="e.g. 149.66,2.99 etc. ", key='amount')
        v15 = st.text_input("Enter V15", placeholder="e.g. 1.34563,-11.648274", key='v15')
        v16 = st.text_input("Enter V16", placeholder="e.g. 1.34563,-11.648274", key='v16')
        v17 = st.text_input("Enter V17", placeholder="e.g. 1.34563,-11.648274", key='v17')
        v18 = st.text_input("Enter V18", placeholder="e.g. 1.34563,-11.648274", key='v18')
        v19 = st.text_input("Enter V19", placeholder="e.g. 1.34563,-11.648274", key='v19')
        v20 = st.text_input("Enter V20", placeholder="e.g. 1.34563,-11.648274", key='v20')
        v21 = st.text_input("Enter V21", placeholder="e.g. 1.34563,-11.648274", key='v21')
        v22 = st.text_input("Enter V22", placeholder="e.g. 1.34563,-11.648274", key='v22')
        v23 = st.text_input("Enter V23", placeholder="e.g. 1.34563,-11.648274", key='v23')
        v24 = st.text_input("Enter V24", placeholder="e.g. 1.34563,-11.648274", key='v24')
        v25 = st.text_input("Enter V25", placeholder="e.g. 1.34563,-11.648274", key='v25')
        v26 = st.text_input("Enter V26", placeholder="e.g. 1.34563,-11.648274", key='v26')
        v27 = st.text_input("Enter V27", placeholder="e.g. 1.34563,-11.648274", key='v27')
        v28 = st.text_input("Enter V28", placeholder="e.g. 1.34563,-11.648274", key='v28')

    if st.button("🔮 Predict Fraud", type="primary"):
        try:
            # Build the input dataframe
            input_data = pd.DataFrame({
                'time': [time],
                'amount': [amount],
                'v1': [v1],
                'v2': [v2],                
                'v3': [v3],
                'v4': [v4],
                'v5': [v5],
                'v6': [v6],
                'v7': [v7],
                'v8': [v8],
                'v9': [v9],
                'v10': [v10],
                'v11': [v11],
                'v12': [v12],
                'v13': [v13],
                'v14': [v14],
                'v15': [v15],
                'v16': [v16],
                'v17': [v17],
                'v18': [v18],
                'v19': [v19],
                'v20': [v20],
                'v21': [v21],
                'v22': [v22],
                'v23': [v23],
                'v24': [v24],
                'v25': [v25],
                'v26': [v26],
                'v27': [v27],
                'v28': [v28]
            })

            input_data = input_data.astype(float)  # 🚨 Important: Convert all to float
            input_data_np = input_data.to_numpy()

            prediction = model.predict(input_data_np)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data_np)[0]
                fraud_prob = prob[1] * 100  # Probability of fraud
                not_fraud_prob = prob[0] * 100
            else:
                fraud_prob = None
                not_fraud_prob = None

            st.success("✅ Prediction Complete!")
            st.markdown("### 🔍 Prediction Result:")

            if prediction == 1:
                st.markdown(f"<h3 style='color: red;'>🚨 Fraudulent Transaction Detected!</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: green;'>✅ Legitimate Transaction</h3>", unsafe_allow_html=True)

            if fraud_prob is not None:
                st.markdown(f"**🧮 Probability of Fraud:** `{fraud_prob:.2f}%`")
                st.markdown(f"**🔐 Probability of Legitimate:** `{not_fraud_prob:.2f}%`")

        except Exception as e:
            st.error(f"❌ Error: {e}")



# Sidebar with model information
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>ℹ️ Model Information</h1>", unsafe_allow_html=True)
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not loaded")
    
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("1. Fill in all the fields.")
    st.markdown("2. Click 'Predict Credit Card'.")
    st.markdown("3. View the result below.")

