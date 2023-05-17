import streamlit as st
import numpy as np
import pandas as pd
from functions import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import seaborn as sns
import math
from functions import utils, extras




if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Upload Data` page!")
else:
    st.markdown("## Exploratory Data Analysis")
    data = pd.read_csv('data/main_data.csv')

    # Read the column meta data for this dataset 
    col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

    ''' The EDA section allows you to explore your dataset and gain insights 
        into its characteristics. Here, you can view basic statistics, 
        visualize the distribution of the data, and investigate relationships 
        between variables.
    '''
    st.markdown("#### Get insights of your data and visualize it")
    



if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Upload Data` page!")
else:
    df_analysis = pd.read_csv('data/main_data.csv')
    df_visual = df_analysis.copy()
    cols = pd.read_csv('data/metadata/column_type_desc.csv')
    Categorical,Numerical,Object = utils.getColumnTypes(cols)
    cat_groups = {}
    unique_Category_val={}
    
    unique_Category_val = {}
    for i in range(len(Categorical)):
        unique_Category_val[Categorical[i]] = utils.mapunique(df_analysis, Categorical[i])
        cat_groups[Categorical[i]] = {Categorical[i]: df_visual.groupby(Categorical[i])}

            
    
    category = st.selectbox("Select Category ", Categorical + Object)

    sizes = (df_visual[category].value_counts()/df_visual[category].count())

    labels = sizes.keys()



    
    
    palette = cm.get_cmap('GnBu')

    # Create a list of colors using the colormap
    colors = palette(np.linspace(0, 1, len(labels)))

    # Create a new figure with a pie chart
    fig, ax = plt.subplots(facecolor='w')
    ax.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=0, colors=colors)

    # Ensure the pie chart is circular
    ax.axis('equal')

    # Add a title to the plot
    title = "Distribution of '{}'".format(str(category))
    ax.set_title(title, fontsize=14)

    # Display the plot using Streamlit
    st.pyplot(fig)    
    

    
    corr = df_analysis.corr()
    

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Create a heatmap of the correlation matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    

    sns.heatmap(corr, mask=mask, cmap='GnBu', annot=True, fmt='.2f', square=True, ax=ax)

    # Add a title to the plot
    ax.set_title('Correlation Matrix Heatmap',  fontsize=14)

    # Display the plot using Streamlit
    st.pyplot(fig)
    
    

    
    categoryObject = st.selectbox("Select " + (str)(category),unique_Category_val[category])
    st.write(cat_groups[category][category].get_group(categoryObject).describe())

    
    colName = st.selectbox("Select Column ",Numerical)

    

    
    
        
    # Create a dictionary of unique category values and corresponding colors
    unique_Category_val = {}
    colors = plt.cm.get_cmap('tab20').colors
    for i, category_val in enumerate(df_visual[category].unique()):
        unique_Category_val[category_val] = colors[i % len(colors)]

    # Create a scatter plot of category vs colName
    fig, ax = plt.subplots()
    for category_val, color in unique_Category_val.items():
        ax.scatter(df_visual[df_visual[category] == category_val][category], 
                df_visual[df_visual[category] == category_val][colName], 
                color=color, label=category_val)

    # Set axis labels and title
    x_label = df_visual[category].name
    y_label = df_visual[colName].name
    title = f'Scatter Plot of {x_label} vs {y_label}'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Display the plot using Streamlit
    st.pyplot(fig)




    
    # Create bins for the numerical variable
    n = len(df_visual[colName])
    bin_size = int(2 * math.pow(n, 1/3))
    bins = pd.cut(df_visual[colName], bins=bin_size)

    # Create a bar chart of the distribution of the bins
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=bins, data=df_visual, ax=ax, palette="GnBu")

    # Set axis labels and title
    x_label = df_visual[colName].name.title() + " (Bins)"
    y_label = "Count"
    title = f"Distribution of {df_visual[colName].name.title()}"

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=18)

    # Rotate x-axis tick labels vertically
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Remove the top and right spines
    sns.despine()

    # Display the plot using Streamlit
    st.pyplot(fig)



    # Add a button to navigate to another page
    if st.button("Next"):
        extras.switch_page("Train")  
    
    
    #st.bar_chart(cat_groups[category][category].get_group(categoryObject)[colName])
    #st.bar_chart(df_visual[colName])
    #st.bar_chart(df_visual[colName].sort_values(ascending=False))    
