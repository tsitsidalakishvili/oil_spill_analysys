import streamlit as st
import pandas as pd
#from ydata_profiling import ProfileReport
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import time
from sklearn.model_selection import train_test_split  # Add this import statement
from sklearn.impute import SimpleImputer
import io

import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Add this import statement
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt



# Set page config
st.set_page_config(page_title='Oil spill analysis', page_icon='sensor', layout="wide")
st.title('Oil spill analysis')

# Create tabs
tabs = ["Oil spill analysis", "ML Model Building"]
selected_tab = st.sidebar.radio("Select a tab:", tabs)

# Oil spill analysis tab
if selected_tab == "Oil spill analysis":
    # Provide information about the app
    with st.expander('About this app'):
        st.write("""
            **What can this app do?**
            - Build a machine learning model to predict various outcomes based on oil well operation parameters.
            - The user can upload a CSV file for analysis and generate a comprehensive profile report.
            
            **Use Case Example**
            - Predict future 'Oil volume (m3/day)' to plan production using data.
            
            Libraries used:
            - Pandas, NumPy for data handling
            - Scikit-learn for machine learning
            - Streamlit and ydata-profiling for the web app and data analysis
            - Plotly for data visualization
            - Matplotlib for additional visualizations
        """)

    # Upload CSV file

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if not df.empty:
            st.write("Uploaded CSV file:")
            st.dataframe(df)  # Display the uploaded data as a table
            
  

            # NEW: Display initial records and dataset information
            with st.expander("Initial Records in Dataset"):
                st.dataframe(df.head())

            with st.expander("Exploring the Dataset"):
                # Shape of the dataframe
                st.write("Shape of the dataframe:", df.shape)

                # Getting more info on the dataset
                st.text("Info on the dataset:")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

                # Describe dataset
                st.write("Statistical summary of the dataset:")
                st.dataframe(df.describe())

                # Display missing values count
                st.write("Missing values in each column:")
                missing_values = df.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                st.table(missing_values)



            with st.expander("Dealing with Missing Values"):
                st.write("Select how you want to deal with missing values in the dataset.")
                option = st.selectbox("Choose a method:",
                                    ['Select an option', 'Drop rows with missing values', 'Impute with Mean', 'Impute with Median'])

                if option == 'Drop rows with missing values':
                    df.dropna(inplace=True)
                    st.success("Rows with missing values dropped successfully.")
                    st.dataframe(df)

                elif option == 'Impute with Mean':
                    imputer = SimpleImputer(strategy='mean')
                    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
                    df[df.select_dtypes(include=['float64', 'int64']).columns] = df_imputed
                    st.success("Missing values imputed with column mean successfully.")
                    st.dataframe(df)

                elif option == 'Impute with Median':
                    imputer = SimpleImputer(strategy='median')
                    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
                    df[df.select_dtypes(include=['float64', 'int64']).columns] = df_imputed
                    st.success("Missing values imputed with column median successfully.")
                    st.dataframe(df)


            # Column 1
            col1, col2 = st.columns(2)
            
            # Row 1
            with col1:
                # Chart 1: Cause of Oil Spillage
                main_causes = df['Cause Category'].value_counts()
                fig1 = go.Figure(go.Bar(x=main_causes.index, y=main_causes.values))
                fig1.update_layout(title='Cause of Oil Spillage')
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Chart 2: Cause Sub Category
                main_causes_sub = df['Cause Subcategory'].value_counts()
                fig2 = go.Figure(go.Bar(x=main_causes_sub.index, y=main_causes_sub.values))
                fig2.update_layout(title='Cause Sub Category')
                st.plotly_chart(fig2, use_container_width=True)

            # Row 2
            col3, col4 = st.columns(2)
            with col3:
                # Chart 3: Count Plot for Pipeline Type
                fig3 = px.bar(df, x='Pipeline Type', title='Count of Pipeline Types')
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                # Chart 4: Bar Plot for Costs by Pipeline Type
                oil_pipelinetype_cost = df[['Pipeline Type', 'All Costs']]
                fig4 = px.bar(oil_pipelinetype_cost, x='Pipeline Type', y='All Costs', title='Costs by Pipeline Type')
                st.plotly_chart(fig4, use_container_width=True)

            # Row 3
            col5, col6 = st.columns(2)
            with col5:
                # Chart 5: Year-by-Year Cost Distributions
                df['All Costs'] /= 1e6  # Convert costs to millions for readability
                df_filtered = df[df['Accident Year'] != 2017]  # Example of filtering out a specific year
                cost_columns = ['Property Damage Costs', 'Lost Commodity Costs', 'Public/Private Property Damage Costs', 'Emergency Response Costs', 'Environmental Remediation Costs']
                df_sum = df_filtered.groupby('Accident Year')[cost_columns + ['All Costs']].sum()
                df_mean = df_filtered.groupby('Accident Year')[cost_columns + ['All Costs']].mean()

                # Create subplots
                fig5 = make_subplots(rows=1, cols=2, subplot_titles=("Sum of Costs by Year", "Fraction of Total Cost by Year"))
                
                # Sum of costs plot
                for col in cost_columns:
                    fig5.add_trace(go.Scatter(x=df_sum.index, y=df_sum[col], mode='lines+markers', name=col), row=1, col=1)
                
                # Fraction of total cost plot
                for col in cost_columns:
                    fig5.add_trace(go.Scatter(x=df_mean.index, y=df_mean[col]/df_mean['All Costs'], mode='lines+markers', name=col), row=1, col=2)
                
                fig5.update_layout(height=600, width=800, title_text="Year-by-Year Cost Distributions")
                st.plotly_chart(fig5, use_container_width=True)

    
    

            with col6:
                # Chart 6: Most Frequent Oil Spillers 
                CCC = Counter(df['Operator ID'].values)
                ids = [x[0] for x in CCC.most_common(20)]

                hXvalue = []
                hValues = []
                hLabels = []
                for j, i in enumerate(ids):
                    hXvalue.append(j+1)
                    hValues.append(CCC[i])
                    hLabels.append(" or ".join(df['Operator Name'].loc[df['Operator ID'] == i].unique()))

                fig6 = plt.figure(figsize=(7, 7))
                plt.barh(hXvalue[::-1], hValues, align='center', color='forestgreen')
                plt.ylim([0, len(hValues)+1])
                plt.title('Top 20 Most Frequent \'Spillers\' (Operator)', fontsize=20, y=1.04)
                plt.yticks(hXvalue[::-1], hLabels, fontsize=10)
                plt.xlabel('Number of Spills', fontsize=16)
                plt.xticks(fontsize=10)
                st.pyplot(fig6)
    
    
            # Add a new expander for the map plotting
            # Determine the most frequent spillers
            most_frequent_spillers = df['Operator ID'].value_counts().nlargest(20).index
            df_top_spillers = df[df['Operator ID'].isin(most_frequent_spillers)]

            # Layout for the Most Frequent Oil Spillers chart
            st.write("Top 20 Most Frequent Spillers")
            top_spiller_counts = df_top_spillers['Operator ID'].value_counts()
            fig6 = px.bar(top_spiller_counts, x=top_spiller_counts.values, y=top_spiller_counts.index, orientation='h', labels={'y': 'Operator ID'})
            fig6.update_layout(title='Most Frequent Oil Spillers (Bar Chart)')
            st.plotly_chart(fig6, use_container_width=True)

            # Map of accidents by the most frequent spillers under the bar chart
            st.write("Locations of Accidents by Most Frequent Spillers (Map)")
            fig_map = px.scatter_mapbox(
                df_top_spillers, 
                lat='Accident Latitude', 
                lon='Accident Longitude', 
                color='Operator ID',
                hover_name='Accident City', 
                hover_data=['Accident County', 'Accident State', 'Operator ID'], 
                zoom=3, 
                height=600
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)


