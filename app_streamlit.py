import streamlit as st
import joblib 

model = joblib.load('HB_xgboost_optimized_model.joblib')

st.title("花棒种子预测模型")

feature_names = [f"特征{i+1}" for i in range(29)]  

user_inputs = []  
cols = st.columns(3)

for i, name in enumerate(feature_names):
    col = cols[i % 3]
    value = col.number_input(
        label=name,
        value=0.0,
        key=f"input_{i}"   
    )
    user_inputs.append(value)

if st.button("开始预测"):
    input_data = [user_inputs]   # shape (1,29)
    prediction = model.predict(input_data)[0]  
    st.success(f"预测结果是：{prediction}")