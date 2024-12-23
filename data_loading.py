import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import json


credentials_info = st.secrets["gcp_service_account"]["service_account_info"]
credentials_dict = json.loads(credentials_info)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

client_bq = bigquery.Client(credentials=credentials)

def get_monthly_traffic():
    query = """
    SELECT 
        date,
        COUNT(*) AS total_records,
        SUM(page_views) AS total_page_views,
        SUM(sessions) AS total_sessions,
        SUM(total_users) AS total_users,
        SUM(new_users) AS total_new_users,
        SUM(user_engagement_duration) AS total_engagement_duration
    FROM 
        `pr-project-444202.GA4_data.traffic_data`
    GROUP BY 
        date
    ORDER BY 
        date DESC;
    """

    # Execute the query
    query_job = client_bq.query(query)
    # Convert query result to DataFrame
    monthly_traffic_data = query_job.to_dataframe()

    return monthly_traffic_data

def fetch_bigquery_table(table_id):
    """
    Fetches the entire contents of a BigQuery table into a pandas DataFrame.

    Parameters:
    - project_id (str): The GCP project ID.
    - table_id (str): The BigQuery table ID in the format 'project.dataset.table'.

    Returns:
    - pandas.DataFrame: The table contents as a DataFrame.
    """
    # Create a BigQuery client
    client = bigquery.Client()

    # Query to select all data from the table
    query = f'''SELECT * FROM `{table_id}`'''

    # Run the query and return the results as a DataFrame
    dataframe = client.query(query).result().to_dataframe()
    return dataframe

monthly_traffic_data = get_monthly_traffic()
st.write(monthly_traffic_data)
