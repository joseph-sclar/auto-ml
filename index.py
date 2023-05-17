import streamlit as st
from functions import extras



# Set page header
st.title("Welcome to SmartAI")

# Add a description
st.write("""
         SmartAI is an easy-to-use tool that lets you explore and train machine learning models with your own clean data, without any coding experience required. Whether you're a data scientist, business analyst, or just curious about machine learning, SmartAI is designed to help you get started quickly and easily.

Simply upload your data to SmartAI, select the features you want to analyze, and start exploring. With powerful visualization tools and pre-built machine learning models, SmartAI makes it easy to gain insights into your data and make predictions. You don't need any programming experience to use it.

With SmartAI, you can explore your data, visualize it, and train machine learning models without any hassle. Whether you're looking to make predictions, identify patterns, or gain insights into your data, SmartAI has everything you need to get started with machine learning."
""")

# Add a section header
st.header("Getting Started")

# Add a button to navigate to another page
if st.button("Get Started"):
    extras.switch_page("Data")

# Add a section header
st.header("About")

# Add some text
st.write("This app was created by Joseph Sclar.")

# Add a section header
st.header("Contact")

# Add an email link
st.write("Email: [joseph.sclar2@gmail.com](mailto:joseph.sclar2@gmail.com)")

# Add a link to the app's GitHub repo
st.write("[GitHub Repo](https://github.com/your_username/your_app)")

