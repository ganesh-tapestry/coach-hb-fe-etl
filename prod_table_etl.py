import pandas as pd

import numpy as np


import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
db_params = {
    'dbname': 'gen_ai_nlp',  # Connect to the 'gen_ai_nlp' database
    'user': os.getenv('DB_USER'),
    'password': urllib.parse.quote_plus(os.getenv('DB_PASSWORD') or ''),  # URL encode the password
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT'))  # Ensure port is read as an integer
}

# Create the connection string
conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"

# Create the database engine
engine = create_engine(conn_str)

# Read the Excel file containing style codes from column B
style_codes_df = pd.read_excel('feature_output 1.xlsx', usecols='B', engine='openpyxl')

# Initialize a list to store the count details and style code
count_details = []

# Iterate over each style code in 

# TODO: write code to get data based on style id




# Sample DataFrame
data = {
    'style_code': ['C0707', 'C0909', 'C0707'],
    'sku_code': ['C0707 IMGRN', 'C0909 IMBLK', 'C0707 IMRED'],
    'nrf_color': [300, 651, 300],
    'article_cat_ind': ['00', '02', '00'],
    'color_code': ['B4F8D', 'B4F8F', 'B4F8D'],
    'size_code': ['11 B', '5 B', '7.5 D'],
    'style_desc': ['Jade-4C perforated wedge', 'Everette Leather Sandal', 'Jade-4C perforated wedge'],
    'gender': ['M', 'F', 'M'],
    'sku_desc': ['JADE, BLACK, 11 B', 'EVERETTE LEATHER, CHK, 5 B', 'JADE, BLACK, 7.5 D'],
    'color_desc': ['BLACK', 'Chalk', 'BLACK'],
    'upc_code': [787934821016, 195031897047, 787934821016]
}


for style_code in style_codes:
        
    # TODO: # sql df load "select (only the colms to be transformed + style id column) from pro_table where country = "USA" and style="C1234"
    sql_df = pd.DataFrame(data)


    op_df = pd.DataFrame()



    op_row = dict()
    # Label Encoding for style_code 
    # we want count of colurs available for the product
    op_row['color_count'] = sql_df['color_code'].nunique()


    def encode_gender(data):
        if data == "M":
            return "M"
        elif data == "F":
            return "F"
        elif data == "AGD":
            return "A"
        else:
            None

        
    op_row['gender'] = sql_df['gender'].apply(lambda x: encode_gender(x),axis=1)






    # Concatenation for sku_code
    df['sku_identifier'] = df['style_code'] + ' ' + df['color_code']

    # Label Encoding for nrf_color
    df['nrf_color_count'] = df['nrf_color'].astype('category').cat.codes

    # Encoding for article_cat_ind
    df['article_category_index'] = np.where(df['color_code'] == '0', 0, 1)

    # Encoding for color_code
    df['colors_counts'] = df.groupby('style_code')['color_code'].transform('nunique')

    # One-hot Encoding for size_code
    df = pd.concat([df, pd.get_dummies(df['size_code'], prefix='size')], axis=1)

    # Text Normalization for style_desc
    df['product_style_description'] = df['style_desc'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)

    # Categorical Encoding for gender
    df['gender_encoded'] = df['gender']

    # String Concatenation for sku_desc
    df['sku_description'] = df['sku_desc']

    # Label Encoding for color_desc
    df['product_color_description'] = df['color_desc'].astype('category').cat.codes

    # Direct Usage for upc_code
    df['upc_identifier'] = df['upc_code']

    print(df)
    
    op_df.append(op_row)
