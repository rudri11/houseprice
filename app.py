import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

st.set_page_config(page_icon=":house:",page_title="House Price Prediction")
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
        <style>
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }
         .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    choose = option_menu("App Gallery", ["About","DataSet","Data Analysis","Data Visualization","Prediction","About Internship"],
                         icons=['house','database-fill-add', 'activity ', 'bar-chart-fill','gear-wide','filetype-py'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        # "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def About():
    st.title("House Price Prediction App")
    st.write("""
    Welcome to the House Price Prediction App! This app predicts the price of houses based on various features.
    """)

    st.header("About")
    st.write("""
    This web application is designed for educational purposes to demonstrate how machine learning models can predict house prices based on features like location, number of rooms, etc.
    """)

    st.header("How to Use")
    st.write("""
    1. Enter the necessary details about the house.
    2. Click on the 'Predict' button to get the predicted house price.
    3. The model will then display the predicted price based on the input features.
    """)

    st.header("Dataset")
    st.write("""
    The dataset used for training the model contains information about various houses including their prices and features such as number of bedrooms, size of the house, location, etc.
    """)

    st.header("Technologies Used")
    st.write("""
    This project utilizes Python, Streamlit for building the web app, and a machine learning model trained using libraries like Scikit-learn.
    """)

    st.header("Developers")
    st.write("""
    - INFOLABZ IT SERVICES PVT. LTD.
    """)

    st.header("Contact")
    st.write("""
    For any inquiries, please contact [developers@infolabz.in].
    """)

def data_analysis():
    df = pd.read_csv('house_price_prediction_dataset.csv')

        # Streamlit interface


    st.title('House Price Prediction Data Analysis')

    # Load and display raw data
    st.header("Raw Data")
    st.write(df)

    # Clean the data (remove unnecessary columns)
    columns_to_drop = ["proximity_to_city_center", "neighborhood_quality", "lot_size"]
    cleaned_df = df.drop(columns=columns_to_drop)

    # Display cleaned data
    st.header("Cleaned Data (After Removing Columns)")
    st.write(cleaned_df)

    # Train-test split
    X = cleaned_df.drop(columns="house_price")
    y = cleaned_df["house_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display train-test split information
    st.header("Train-Test Split")
    st.write("Training Set:")
    st.write(X_train)
    st.write("Testing Set:")
    st.write(X_test)
def data_visualization():
    df = pd.read_csv('house_price_prediction_dataset.csv')

    st.title('House Price Prediction Data Analysis')

    # Load and display raw data
    st.header("Raw Data")
    st.write(df)

    # Clean the data (remove unnecessary columns)
    columns_to_drop = ["proximity_to_city_center", "neighborhood_quality", "lot_size"]
    cleaned_df = df.drop(columns=columns_to_drop)

    # Display cleaned data
    st.header("Cleaned Data (After Removing Columns)")
    st.write(cleaned_df)

    # Data analysis with graphs and observations
    st.header("Data Analysis with Graphs")

    # Select numerical columns for visualization
    numerical_columns = ['num_bedrooms', 'num_bathrooms', 'square_footage', 'age_of_house']

    # Create histograms and write observations
    for col in numerical_columns:
        st.subheader(f"Histogram of {col}")
        plt.figure(figsize=(8, 6))
        sns.histplot(cleaned_df[col], bins=20, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        st.pyplot()

        st.subheader(f"Observations for {col}")
        st.write(f"Mean {col}: {cleaned_df[col].mean()}")
        st.write(f"Median {col}: {cleaned_df[col].median()}")
        st.write(f"Min {col}: {cleaned_df[col].min()}")
        st.write(f"Max {col}: {cleaned_df[col].max()}")
        st.write(f"Standard Deviation {col}: {cleaned_df[col].std()}")

    # Correlation heatmap for numerical features
    st.header("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cleaned_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()


def prediction():
    df = pd.read_csv('house_price_prediction_dataset.csv')
    df = df.drop(columns=["proximity_to_city_center", "neighborhood_quality", "lot_size"])

    X = df.drop(columns="house_price")
    y = df["house_price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Linear Regression model
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = mlr.predict(X_test)

    # Calculate the R^2 score
    r2 = r2_score(y_test, predictions)
    accuracy = r2 * 100

    # Streamlit interface
    st.title('House Price Prediction')

    # st.write(f'R^2 Score: {r2}')
    # st.write(f'Accuracy: {accuracy:.2f}%')

    num_bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, max_value=10, value=3)
    num_bathrooms = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=5, value=2)
    square_footage = st.number_input("Enter the square footage:", min_value=300, max_value=10000, value=1500)
    age_of_house = st.number_input("Enter the age of the house:", min_value=0, max_value=150, value=20)

    user_input = pd.DataFrame({
        'num_bedrooms': [num_bedrooms],
        'num_bathrooms': [num_bathrooms],
        'square_footage': [square_footage],
        'age_of_house': [age_of_house]
    })

    if st.button('Predict House Price'):
        predicted_price = mlr.predict(user_input)
        st.header(f'Predicted House Price: ${predicted_price[0]:,.2f}')

def internship():

    st.title(' Student Details')


    # Student Details
    st.header('Student Details')
    st.write('Student Name: Rudri')
    st.write('Program: Computer Science Engineering')
    st.write('University: Gujarat Technological University')





if choose == "About":
    About()

elif choose == "Data Analysis":
    data_analysis()
elif choose == "Data Visualization":
    data_visualization()
elif choose == "Prediction":
    prediction()

elif choose == "DataSet":
    df = pd.read_csv('house_price_prediction_dataset.csv')
    st.title("DATAFRAME")
    st.dataframe(df)
elif choose == "About Internship":
    internship()