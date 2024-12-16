import streamlit as st
import pandas as pd
import joblib
import os

# Application Title
st.title("Stock Return Predictor Using GDP Data")

# File Upload Section
uploaded_file = st.file_uploader("Upload the PKL file containing trained models:", type=['pkl'])

if uploaded_file:
    try:
        # Load the overall results from the uploaded PKL file
        overall_results = joblib.load(uploaded_file)
        
        # Show the stocks available in the uploaded PKL file
        stock_list = [result['stock'] for result in overall_results]
        selected_stock = st.selectbox("Select a Stock:", stock_list)

        # Load GDP Data for column selection (assumed GDP data as a placeholder)
        gdp_file = st.file_uploader("Upload GDP Data (Excel file):", type=['xlsx'])

        if gdp_file:
            # Load GDP data to get column names
            gdp_data = pd.read_excel(gdp_file, engine='openpyxl')
            gdp_columns = gdp_data.columns.tolist()
            
            # Display available columns and let the user select inputs
            selected_columns = st.multiselect("Select GDP-related columns for prediction:", gdp_columns, default=['GDP', 'Inflation', 'Interest Rate'])

            if selected_columns:
                # Display input fields for upcoming GDP values
                st.subheader("Input Upcoming Values for GDP Data:")
                upcoming_values = {}
                for column in selected_columns:
                    upcoming_values[column] = st.number_input(f"Enter value for {column}:")

                # Predict button
                if st.button("Predict Stock Returns"):
                    # Find the model for the selected stock
                    model_result = next(result for result in overall_results if result['stock'] == selected_stock)
                    model = model_result['model']

                    # Prepare the input DataFrame
                    input_data = pd.DataFrame([upcoming_values])

                    # Predict using the selected model
                    predicted_return = model.predict(input_data)[0]

                    # Display the prediction
                    st.success(f"Expected Return for {selected_stock}: {predicted_return:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
