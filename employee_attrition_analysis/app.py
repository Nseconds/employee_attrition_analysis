import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the pre-trained model
model = joblib.load('employee_attrition_model.joblib')

# Load the dataset
df = pd.read_csv('data/IBM_HR_Analytics_Employee_Attrition.csv')

def main():
    st.title('Employee Attrition Prediction Dashboard')

    # Sidebar for data exploration and prediction
    st.sidebar.header('Explore & Predict')

    # Data Exploration Section
    exploration_option = st.sidebar.selectbox(
        'Select Exploration Type', 
        ['Attribute Relationships', 'Attrition Distribution']
    )

    if exploration_option == 'Attribute Relationships':
        # Attribute Relationship Visualization
        x_attr = st.sidebar.selectbox('Select X-axis Attribute', df.columns)
        y_attr = st.sidebar.selectbox('Select Y-axis Attribute', df.columns)

        fig = px.scatter(
            df, 
            x=x_attr, 
            y=y_attr, 
            color='Attrition', 
            title=f'{x_attr} vs {y_attr} with Attrition'
        )
        st.plotly_chart(fig)

    else:
        # Attrition Distribution by Categorical Variables
        cat_attr = st.sidebar.selectbox('Select Categorical Attribute', 
            ['Department', 'JobRole', 'Gender', 'MaritalStatus']
        )

        attrition_dist = df.groupby([cat_attr, 'Attrition']).size().unstack(fill_value=0)
        fig = px.bar(
            attrition_dist, 
            title=f'Attrition Distribution by {cat_attr}'
        )
        st.plotly_chart(fig)

    # Prediction Section
    st.header('Attrition Prediction')

    # Input features for prediction
    prediction_features = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
    'RelationshipSatisfaction', 'WorkLifeBalance', 'YearsSinceLastPromotion',
    'YearsAtCompany', 'PercentSalaryHike', 'TrainingTimesLastYear',
    'TotalWorkingYears', 'StockOptionLevel', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'PerformanceRating'
     ]


    # Create input fields
    input_data = {}
    for feature in prediction_features:
        if df[feature].dtype == 'object':
            input_data[feature] = st.selectbox(f'Select {feature}', df[feature].unique())
        else:
            input_data[feature] = st.slider(
                f'Select {feature}', 
                int(df[feature].min()), int(df[feature].max()), 
                int(df[feature].mean())
            )

    # Prediction button
    if st.button('Predict Attrition'):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display prediction
        st.subheader('Prediction Results')
        if prediction[0] == 1: 
            st.error('This employee is likely to leave.')
        else:
            st.success('This employee is likely to stay.')

if __name__ == '__main__':
    main()
