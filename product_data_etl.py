import pandas as pd
import psycopg2
from psycopg2 import sql

# PostgreSQL Connection Setup
connection = psycopg2.connect(
    dbname="gen_ai_nlp",
    user="standon",
    password="$T@nD0n@N0nPr0d",  # You need to add your password
    host="stitch-rag-knowledgebase-dev-cluster.cluster-ro-c70imsy2ydvn.us-east-1.rds.amazonaws.com",
    port=5432
)

cursor = connection.cursor()

# Function to Process and Dump Encoded Data
def db_dump_product(style_code):
    try:
        # Fetch data for the specific style_code
        query = f"""
        SELECT
            style_code,
            department_desc,
            gender,
            article_cat_ind,
            color_code,
            class,
            subclass,
            collection,
            subcollection,
            licensed,
            material,
            material_type,
            collaboration,
            silhouette,
            style_group,
            active,
            size_code,
            article_type,
            ftystoretype
        FROM gen_ai_nlp.rpt_a360_item_lkp
        WHERE style_code = '{style_code}' and brandcode = 'COH'
        """
        df = pd.read_sql_query(query, connection)

        # If no data is found, return None
        if df.empty:
            print(f"No data found for style_code: {style_code}")
            return None

        # Encoding logic
        df['gender'] = df['gender'].map({'W': 'W', 'M': 'M', 'AGD': 'A', 'U': 'U'}).fillna('Unknown')
        df['article_cat_ind'] = df['color_code'].apply(lambda x: 2 if pd.notnull(x) else 0)
        color_count = df['color_code'].nunique()

        # Replace missing values in features with 'null'
        features = [
            'class', 'subclass', 'collection', 'subcollection', 'licensed',
            'material', 'material_type', 'collaboration', 'silhouette',
            'style_group', 'active', 'size_code', 'department_desc', 'article_type',
            'ftystoretype'
        ]
        for feature in features:
            df[feature] = df[feature].fillna('null')

        # Prepare the final row (single row per style code)
        final_row = df.iloc[0].to_dict()
        final_row['color_count'] = color_count
        final_row['factory_store_type'] = df['ftystoretype'].iloc[0]
        final_row.pop('color_code', None)  # Remove the color_code from the final dump

        # Prepare the data to insert into PostgreSQL
        insert_query = """
        INSERT INTO coh_hb_analysis.coh_product (
            style_code, department_desc, gender, color_count, article_cat_ind, class, subclass,
            collection, subcollection, licensed, material, material_type, collaboration,
            silhouette, style_group, active, size_code, article_type, factory_store_type
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data_to_insert = (
            final_row['style_code'], final_row['department_desc'], final_row['gender'],
            final_row['color_count'], final_row['article_cat_ind'], final_row['class'], final_row['subclass'],
            final_row['collection'], final_row['subcollection'], final_row['licensed'],
            final_row['material'], final_row['material_type'], final_row['collaboration'],
            final_row['silhouette'], final_row['style_group'], final_row['active'],
            final_row['size_code'], final_row['article_type'],
            final_row['factory_store_type']
        )

        # Execute the insert query
        cursor.execute(insert_query, data_to_insert)
        connection.commit()

        print(f"Data for style_code '{style_code}' inserted successfully!")

        # Return the final encoded row
        return final_row

    except Exception as e:
        print(f"Error processing style_code '{style_code}': {e}")
        connection.rollback()
        return None

# Fetch already processed style_codes to avoid duplicates
fetch_processed_query = "SELECT DISTINCT style_code FROM coh_hb_analysis.coh_product"
cursor.execute(fetch_processed_query)
processed_style_codes = {row[0] for row in cursor.fetchall()}

# Fetch all distinct style codes from the source table
fetch_query = "SELECT DISTINCT style_code FROM gen_ai_nlp.rpt_a360_item_lkp WHERE brandcode = 'COH'"
# fetch_query = """
# SELECT DISTINCT style_code 
# FROM gen_ai_nlp.rpt_a360_item_lkp 
# WHERE brandcode = 'COH' 
# AND gender = 'W' 
# """
# AND department_desc = 'Womens Bags'
# """

cursor.execute(fetch_query)
all_style_codes = {row[0] for row in cursor.fetchall()}

# Identify new style_codes that need to be processed
new_style_codes = all_style_codes - processed_style_codes

print(f"Total style codes: {len(all_style_codes)}, Already processed: {len(processed_style_codes)}, New to process: {len(new_style_codes)}")

# Process each new style code dynamically
for code in new_style_codes:
    encoded_row = db_dump_product(code)
    if encoded_row:
        print(f"Processed Row for Style Code {code}: {encoded_row}")
    else:
        print(f"No data found for Style Code {code}")

# Close the PostgreSQL connection
cursor.close()
connection.close()
