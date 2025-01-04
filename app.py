import gradio as gr
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
import joblib

#  pre-trained models
bgf_model = joblib.load("bgf.pkl") 
gamma_gamma_model = joblib.load("ggf.pkl")

# Preload the uploaded CSV file
csv_file_path = "output.csv"  
data = pd.read_csv(csv_file_path) 

# Process RFM and T for the uploaded data
def calculate_rfm():
    try:
        # Convert InvoiceDate to datetime format
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

        # Ensure Total column exists
        if "Total" not in data.columns:
            data["Total"] = data["Quantity"] * data["UnitPrice"]

        # Rename columns for compatibility with Lifetimes
        data.rename(columns={"CustomerID": "customer_id", "InvoiceDate": "date"}, inplace=True)

        # Calculate RFM  using Lifetimes
        rfm = summary_data_from_transaction_data(
            data,
            customer_id_col="customer_id",
            datetime_col="date",
            monetary_value_col="Total",
            observation_period_end=data["date"].max()
        )
        rfm.reset_index(inplace=True)
        rfm.rename(columns={
            "customer_id": "CustomerID",
            "recency": "Recency",
            "frequency": "Frequency",
            "T": "T",
            "monetary_value": "Monetary Value"
        }, inplace=True)

        # Filter rows with valid values
        rfm = rfm[(rfm["Recency"] > 0) & (rfm["Frequency"] > 0) & (rfm["T"] > 0) & (rfm["Monetary Value"] > 0)]

        # Return only the first five rows
        return rfm[["CustomerID", "Recency", "Frequency", "T", "Monetary Value"]].head(5)
    except Exception as e:
        return f"Error processing RFM: {str(e)}"

# Predict purchases and revenue for a specific customer
def predict_customer(customer_id, time_horizon):
    try:
        # Fetch customer RFM
        customer_row = rfm_data[rfm_data["CustomerID"] == int(customer_id)].iloc[0]
        recency = customer_row["Recency"]
        frequency = customer_row["Frequency"]
        T = customer_row["T"]
        monetary_value = customer_row["Monetary Value"]

        # Predict purchases
        predicted_purchases = bgf_model.conditional_expected_number_of_purchases_up_to_time(
            t=time_horizon, frequency=frequency, recency=recency, T=T
        )

        # Predict revenue
        if frequency < 1:
            predicted_revenue = "Not applicable (frequency < 1)"
        else:
            predicted_revenue = gamma_gamma_model.customer_lifetime_value(
                bgf_model,
                frequency=pd.Series([frequency]),
                recency=pd.Series([recency]),
                T=pd.Series([T]),
                monetary_value=pd.Series([monetary_value]),
                time=time_horizon
            ).iloc[0]

        # Format results for better presentation
        html_result = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.5; color: #333;">
            <h3 style="color: #4CAF50;">Customer ID: {customer_id}</h3>
            <p><strong>Recency:</strong> {recency} days</p>
            <p><strong>Frequency:</strong> {frequency} transactions</p>
            <p><strong>T (Tenure):</strong> {T} days</p>
            <p><strong>Monetary Value:</strong> ${monetary_value:.2f}</p>
            <h4 style="color: #2196F3;">Predicted Results</h4>
            <p><strong>Predicted Purchases:</strong> {predicted_purchases:.2f}</p>
            <p><strong>Predicted Revenue:</strong> ${predicted_revenue:.2f}</p>
        </div>
        """
        return html_result
    except Exception as e:
        return f"Error predicting customer: {str(e)}"

# Process the uploaded data to calculate RFM
rfm_data = calculate_rfm()

# Gradio interface
def create_gradio_interface():
    # Interface to display RFM
    display_rfm_interface = gr.Interface(
        fn=lambda: rfm_data, 
        inputs=None,
        outputs=gr.Dataframe(label="RFM Metrics (First 5 Customers)"),
        title="RFM Calculation"
    )

    # Interface for customer-specific predictions
    customer_prediction_interface = gr.Interface(
        fn=predict_customer,
        inputs=[
            gr.Textbox(label="CustomerID"),
            gr.Number(label="Time Horizon (days)")
        ],
        # Updated to HTML for better formatting
        outputs=gr.HTML(label="Prediction Results"),  
        title="Customer Prediction"
    )

    # TODO Gradio 
    # Combine the two interfaces into tabs
    return gr.TabbedInterface(
        [display_rfm_interface, customer_prediction_interface],
        ["View RFM Metrics", "Predict Customer Behavior"]
    )


# Launch the Gradio app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
