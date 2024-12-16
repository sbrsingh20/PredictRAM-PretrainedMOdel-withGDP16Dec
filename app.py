import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Application Title
st.title("Stock Return Predictor Using GDP Data")

# File Upload Section
uploaded_model_file = st.file_uploader("Upload the PKL file containing trained models:", type=['pkl'])

if uploaded_model_file:
    try:
        # Load the overall results from the uploaded PKL file
        overall_results = joblib.load(uploaded_model_file)

        # Extract list of stocks from the loaded model data
        stock_list = [result['stock'] for result in overall_results]
        
        # Select Stock
        selected_stock = st.selectbox("Select a Stock:", stock_list)

        # Upload GDP Data (Excel file)
        gdp_file = st.file_uploader("Upload GDP Data (Excel file):", type=['xlsx'])

        if gdp_file:
            # Load the GDP data to get columns
            gdp_data = pd.read_excel(gdp_file, engine='openpyxl')

            # Display available columns and let the user select inputs
            gdp_columns = gdp_data.columns.tolist()
            selected_columns = st.multiselect("Select GDP-related columns for prediction:", gdp_columns, default=['GDP', 'Inflation', 'Interest Rate'])

            if selected_columns:
                # Display input fields for the selected columns
                st.subheader("Input Upcoming Values for GDP Data:")
                upcoming_values = {}
                for column in selected_columns:
                    upcoming_values[column] = st.number_input(f"Enter value for {column}:")

                # Predict button
                if st.button("Predict Stock Returns"):
                    # Find the model for the selected stock
                    model_result = next(result for result in overall_results if result['stock'] == selected_stock)
                    model = model_result['model']

                    # Prepare the input data for prediction (upcoming GDP values)
                    input_data = pd.DataFrame([upcoming_values])

                    # Ensure the model is using the correct pre-processing steps
                    if isinstance(model, Pipeline):
                        # Use the pipeline to predict
                        predicted_return = model.predict(input_data)[0]
                    else:
                        st.error("The model doesn't have the correct pipeline format!")
                        predicted_return = None

                    # Display the prediction
                    if predicted_return is not None:
                        st.success(f"Expected Return for {selected_stock}: {predicted_return:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
