# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv("dataset/Student_Performance.csv")

# Sample data for demonstration (remove this line when using your own data)
data = data.head(100)

# Add a sidebar
st.sidebar.title('Sidebar')

# Add buttons to the sidebar for selecting options
option1_button = st.sidebar.button('Descriptive Analytics')
option2_button = st.sidebar.button('Excel Regression')
option3_button = st.sidebar.button('Python Regression')

# Determine the selected option based on which button is clicked
if option1_button:
    selected_option = 'Data Set'
elif option2_button:
    selected_option = 'Excel Regression'
elif option3_button:
    selected_option = 'Python Regression'
else:
    selected_option = None

# Display the selected option
if selected_option == 'Data Set':
    # Add code for the "Data Set" option
    st.write("You selected Data Set. Add your data set related code here.")
    st.header("Descriptive Analytics")

    #Pie Chart
    # Create a new column 'Pass/Fail' based on the threshold
    data['Pass/Fail'] = data['Performance Index'].apply(lambda x: 'Passed' if x >= 60 else 'Failed')

    # Count the number of students in each category
    pass_fail_counts = data['Pass/Fail'].value_counts()

    # Create a pie chart
    pie_fig = px.pie(names=pass_fail_counts.index, values=pass_fail_counts.values,
                    title='Distribution of Passed and Failed Students')
    st.plotly_chart(pie_fig)

    # Create the scatter plot
    scatter_fig = px.scatter(data, x='Previous Scores', y='Performance Index', title='Scatter Plot of Previous Scores vs. Performance Index')

    # Show the scatter plot in Streamlit
    st.plotly_chart(scatter_fig)

    # Box Plot
    box_fig = px.box(data, x='Hours Studied', y='Performance Index', title='Box Plot of Hours Studied vs. Performance Index')
    st.plotly_chart(box_fig)
elif selected_option == 'Excel Regression':
    # Add code for the "Excel Regression" option
    st.write("You selected Excel Regression. Add your Excel regression code here.")
    image_path1 = "images/HoursStudied_PreviousScores.png"
    st.image(image_path1, caption='Your Image Caption', use_column_width=True)

    image_path2 = "images/PreviousScores_Sample Question.png"
    st.image(image_path2, caption='Your Image Caption', use_column_width=True)

elif selected_option == 'Python Regression':
    # Add code for the "Python Regression" option
    st.write("You selected Python Regression. Add your Python regression code here.")
    st.header("Python Regression")


    # Define the independent variables
    X3 = data['Previous Scores']
    X4 = data['Sample Question Papers Practiced']
    y = data['Performance Index']

    # Create a DataFrame for regression
    regression_data = pd.DataFrame({'Previous Scores': X3,
                                    'Sample Question Papers Practiced': X4,
                                    'Performance Index': y})

    # Perform linear regression
    model = LinearRegression()
    model.fit(regression_data[['Previous Scores', 'Sample Question Papers Practiced']], y)
    coefficients = model.coef_
    intercept = model.intercept_

    # Create a regression formula
    regression_formula = f"Performance Index = {intercept:.2f} + " \
                        f"{coefficients[0]:.2f} * Previous Scores + " \
                        f"{coefficients[1]:.2f} * Sample Question Papers Practiced"

    # Create a 3D scatterplot
    fig = px.scatter_3d(regression_data, x='Previous Scores', y='Sample Question Papers Practiced', z='Performance Index',
                        opacity=0.7, title='3D Regression Plot')
    fig.update_traces(marker=dict(size=5))

    # Create regression planes
    x3_plane = np.linspace(X3.min(), X3.max(), 50)
    x4_plane = np.linspace(X4.min(), X4.max(), 50)
    x3_plane, x4_plane = np.meshgrid(x3_plane, x4_plane)
    z_plane = intercept + coefficients[0] * x3_plane + coefficients[1] * x4_plane

    fig.add_trace(go.Surface(z=z_plane, x=x3_plane, y=x4_plane, colorscale='Viridis', opacity=0.7))

    # Add regression formula annotation
    formula_annotation = dict(
        text=regression_formula,
        x=0.05,
        y=0.9,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
    )
    fig.update_layout(annotations=[formula_annotation])

    # Show the plot
    fig.update_layout(scene=dict(xaxis_title='Previous Scores',
                                yaxis_title='Sample Question Papers Practiced',
                                zaxis_title='Performance Index'
                                ),
                    width=700,
                    height=900
    )

    st.write("keisamae")
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    #Second plot
    # Define the independent variables
    X3 = data['Hours Studied']
    X4 = data['Previous Scores']
    y = data['Performance Index']

    # Create a DataFrame for regression
    regression_data = pd.DataFrame({'Hours Studied': X3,
                                    'Previous Scores': X4,
                                    'Performance Index': y})

    # Perform linear regression
    model = LinearRegression()
    model.fit(regression_data[['Hours Studied', 'Previous Scores']], y)
    coefficients = model.coef_
    intercept = model.intercept_

    # Create a regression formula
    regression_formula = f"Performance Index = {intercept:.2f} + " \
                        f"{coefficients[0]:.2f} * Hours Studied + " \
                        f"{coefficients[1]:.2f} * Previous Scores"

    # Create a 3D scatterplot
    fig = px.scatter_3d(regression_data, x='Hours Studied', y='Previous Scores', z='Performance Index',
                        opacity=0.7, title='3D Regression Plot')
    fig.update_traces(marker=dict(size=5))

    # Create regression planes
    x3_plane = np.linspace(X3.min(), X3.max(), 50)
    x4_plane = np.linspace(X4.min(), X4.max(), 50)
    x3_plane, x4_plane = np.meshgrid(x3_plane, x4_plane)
    z_plane = intercept + coefficients[0] * x3_plane + coefficients[1] * x4_plane

    fig.add_trace(go.Surface(z=z_plane, x=x3_plane, y=x4_plane, colorscale='Viridis', opacity=0.7))

    # Add regression formula annotation
    formula_annotation = dict(
        text=regression_formula,
        x=0.05,
        y=0.9,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
    )
    fig.update_layout(annotations=[formula_annotation])

    # Show the plot
    fig.update_layout(scene=dict(xaxis_title='Hours Studied',
                                yaxis_title='Previous Scores',
                                zaxis_title='Performance Index'
                                ),
                    width=700,
                    height=900
    )

    st.write("keisamae")
    # Display the plot in Streamlit
    st.plotly_chart(fig)



    

else:
    # Handle the case when no option is selected
    st.header("Midterm Exam Week na!")
