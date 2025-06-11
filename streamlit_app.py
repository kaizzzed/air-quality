import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Air Quality",
    layout="centered",
)

## Step 01 - Setup
st.sidebar.title("Air Quality")
st.sidebar.image("air.jpeg")
page = st.sidebar.selectbox("Select Page",["Introduction","Visualization", "Prediction"])



st.write("   ")
st.write("   ")
st.write("   ")
df = pd.read_csv("Air_Quality.csv")

# CO2 is missing 43056 entries so we dropped it from all visulizations and models
if 'CO2' in df.columns:
    df.drop(['CO2'], axis=1, inplace=True)


## Step 02 - Load dataset
# only include this if you want to turn the city into numbers
dfPrediction = df.copy()
dfPrediction['City_Name'] = dfPrediction['City']  # Keep original names for display
le = LabelEncoder()
dfPrediction['City'] = le.fit_transform(dfPrediction['City'])  # Numerical version for modeling

if page == "Introduction":
    df2 = pd.read_csv("Air_Quality.csv")

    st.subheader("01 Introduction")
    st.markdown("Air pollution causes approximately 7 million premature deaths annually (WHO). This dataset contains records of pollutants and European Air Quality Index (AQI) through January to December 2024 and includes cities from all inhabited continents. Our goal is to predict AQI based on pollutant levels, identify which pollutants most impact AQI, and compare air quality trends across continents.")


    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df2.head(rows))

    st.markdown("##### Missing values")
    missing = df2.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### Summary Statistics")
    if st.toggle("Show Describe Table"):
        st.dataframe(df2.describe())

elif page == "Visualization":

    ## Step 03 - Data Viz
    st.subheader("02 Air Quality Index (AQI) Distribution by City")

    cities = df['City'].unique()
    selected_city = st.selectbox("Select a City", cities)

    # Filter data for the selected city
    city_data = df[df['City'] == selected_city]

    tab1, tab2, tab3 = st.tabs(["Histogram","Bar Chart","Correlation Heatmap"])

    with tab1:
        st.subheader("Distribution of AQI")
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(city_data['AQI'], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of AQI Values in {selected_city}')
        ax.set_xlabel('Air Quality Index (AQI)')
        ax.set_ylabel('Frequency')
        ax.grid(True)

        st.pyplot(fig)

    with tab2:
        st.subheader("Average AQI by Month")

        # Convert 'Date' to datetime and extract month name
        city_data['Date'] = pd.to_datetime(city_data['Date'], errors='coerce')
        city_data = city_data.dropna(subset=['Date'])

        # Extract month names (e.g., Jan, Feb) and create a new column
        city_data['Month'] = city_data['Date'].dt.strftime('%b')  # 'Jan', 'Feb', etc.
        city_data['Month_Num'] = city_data['Date'].dt.month       # Numeric month for sorting

        # Group by Month_Num and Month for average AQI
        monthly_avg = city_data.groupby(['Month_Num', 'Month'])['AQI'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('Month_Num')  # Ensure months are in calendar order

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(monthly_avg['Month'], monthly_avg['AQI'], width=0.6)
        ax.set_title(f"Average Monthly AQI in {selected_city}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average AQI")
        ax.grid(axis='y')

        st.pyplot(fig)


    with tab3:
        st.subheader("City-wise Correlation Matrix of AQI Dataset")
        # Filter and copy data for the selected city
        city_df_corr = df[df['City'] == selected_city].copy()

        # Drop 'City' and 'Date' columns if they exist
        drop_cols = [col for col in ['City', 'Date'] if col in city_df_corr.columns]
        city_df_corr.drop(columns=drop_cols, inplace=True)

        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(city_df_corr.corr().round(2), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title(f'Correlation Matrix for {selected_city}')
        st.pyplot(fig)

elif page == "Prediction":
    st.subheader("03 Prediction with Linear Regression")

    results = {}

    # Train models for each city
    for city_name in dfPrediction['City_Name'].unique():
        city_df = dfPrediction[dfPrediction['City_Name'] == city_name]
        X = city_df.drop(['AQI', 'Date', 'City_Name'], axis=1)
        y = city_df['AQI']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[city_name] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

    # Plotting function
    def plot_city(city_name):
        if city_name not in results:
            return

        y_test = results[city_name]['y_test']
        y_pred = results[city_name]['y_pred']

        plot_df = y_test.copy().to_frame(name='Actual')
        plot_df['Predicted'] = y_pred
        plot_df['Error'] = abs(plot_df['Actual'] - plot_df['Predicted'])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x='Actual',
            y='Predicted',
            hue='Error',
            palette='plasma',
            edgecolor=None,
            legend=True
        )
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title(f'{city_name}: Actual vs Predicted AQI (Hue = Error)')
        plt.plot([plot_df['Actual'].min(), plot_df['Actual'].max()],
                [plot_df['Actual'].min(), plot_df['Actual'].max()],
                '--', color='gray')
        plt.grid(True)
        plt.legend(title='|Error|')
        st.pyplot(plt.gcf())  # Display plot in Streamlit

    # Stats function
    def display_city_stats(city_name):
        if city_name not in results:
            st.write(f"City '{city_name}' not found.")
            return

        stats = results[city_name]
        st.markdown(f"#### Model Performance for {city_name}:")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Mean Squared Error (MSE)", value=f"{stats['MSE']:.2f}")
        col2.metric(label="Mean Absolute Error (MAE)", value=f"{stats['MAE']:.2f}")
        col3.metric(label="R² Score", value=f"{stats['R2']:.3f}")

    # Streamlit tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Brasilia", "Cairo", "Dubai", "London", "New York", "Sydney"])

    with tab1:
        st.subheader("Brasilia")
        plot_city('Brasilia')
        display_city_stats('Brasilia')
        st.markdown("The scatter plot for Brasilia shows a generally strong alignment between actual and predicted AQI values, with most data points clustered near the diagonal, indicating that the linear regression model performs reasonably well. However, the presence of yellow and orange points—especially around mid-range AQI values—highlights areas where prediction errors increase, suggesting the model struggles with moderate pollution levels. The performance metrics reinforce this: a relatively low MSE (23.11) and MAE (3.92) indicate solid accuracy overall, while an R² score of 0.749 shows the model explains about 75% of the variance in AQI data. This suggests Brasilia has moderately predictable pollution patterns, and while the model is useful, more complex approaches may better capture fluctuations at certain AQI levels.")

    with tab2:
        st.subheader("Cairo")
        plot_city('Cairo')
        display_city_stats('Cairo')
        st.markdown("In Cairo’s scatter plot, predicted AQI values show a weak alignment with actual values, with many points deviating significantly from the diagonal, especially at higher AQI levels—indicating substantial prediction errors. The wide spread of yellow and orange points reflects high error magnitudes, suggesting the linear regression model struggles to capture the complexity of pollution patterns in Cairo. The MSE of 142.41 and MAE of 10.21 are both quite high, confirming poor prediction accuracy, while the low R² score of 0.357 shows that the model explains less than 36% of the variance in the data. Overall, this suggests Cairo's pollution levels are highly variable and likely influenced by nonlinear or irregular factors that a simple linear model fails to capture effectively.")

    with tab3:
        st.subheader("Dubai")
        plot_city('Dubai')
        display_city_stats('Dubai')
        st.markdown("In Dubai’s scatter plot, the predicted AQI values diverge significantly from the actual values, with many points clustered far from the diagonal line—indicating widespread and large prediction errors. The concentration of orange and red points (representing high errors) reflects the model’s inability to generalize, especially at higher and lower AQI ranges. This is confirmed by the very high MSE (192.36) and MAE (11.24), indicating severe inaccuracies, while the R² score of -1.108 reveals that the linear regression model performs worse than simply predicting the mean AQI for all observations. Overall, the model is highly ineffective in capturing Dubai's AQI behavior, possibly due to complex, non-linear pollution dynamics or inconsistent patterns that linear regression cannot model.")

    with tab4:
        st.subheader("London")
        plot_city('London')
        display_city_stats('London')
        st.markdown("In London’s plot, the linear regression model performs reasonably well at predicting lower AQI levels, as shown by the dense cluster of purple points near the diagonal. However, as actual AQI increases beyond ~40, the model consistently underpredicts, evident from the growing vertical distance between the points and the diagonal and the shift to orange and yellow hues indicating high error. The MSE of 177.82 and MAE of 8.40 show considerable prediction error, and the low R² score of 0.141 suggests the model explains only 14% of the AQI variability. This indicates that while London’s lower pollution levels are somewhat predictable, the model struggles significantly with higher AQI ranges, likely due to more complex or infrequent pollution events.")

    with tab5:
        st.subheader("New York")
        plot_city('New York')
        display_city_stats('New York')
        st.markdown("In New York’s AQI prediction plot, the model performs fairly well at lower AQI levels (under ~40), as indicated by the tight cluster of purple points near the diagonal. However, it significantly underpredicts AQI for higher actual values (above ~50), where the points shift to red and yellow and fall far below the diagonal—showing large errors. Despite a moderate MAE of 6.96, the MSE of 130.02 and extremely low R² score of 0.042 reveal that the model captures almost none of the underlying AQI variation. This suggests that while New York’s lower pollution levels follow a somewhat consistent pattern, the linear model fails to handle more severe or variable pollution events.")
        
    with tab6:
        st.subheader("Sydney")
        plot_city('Sydney')
        display_city_stats('Sydney')
        st.markdown("Sydney’s AQI prediction plot shows a strong overall fit, with most data points closely following the diagonal line and colored in dark purple, indicating low prediction errors. There is a slight increase in error at mid-range AQI levels (30–50), but the model still tracks the overall trend well. The low MSE (30.51) and MAE (4.21) demonstrate solid accuracy, while the high R² score of 0.756 indicates that the model explains about 76% of the variation in AQI. This suggests that Sydney’s air quality patterns are relatively stable and well-captured by a linear model, reflecting effective environmental monitoring and potentially consistent pollution sources.")
