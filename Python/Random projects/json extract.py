#!/usr/bin/env python
# coding: utf-8

# # Tags json extracts
# ## Author: Bayugo Ayuba Ahmed

# In[ ]:


#Loading libraries
import pandas as pd
import json

user_name = input("Enter you name: ")
print(f"Welcome {user_name}, the purpose of this Python script is to process an Excel file, extract specified fields from a JSON column, remove duplicate rows based on the 'TagNo' column, and save the modified DataFrame to a new Excel file.")
# Prompt the user to enter the file name
file_name = input("Enter the Excel file name (including extension): ")

# Read the Excel file
try:
    df = pd.read_excel(file_name)
except FileNotFoundError:
    print(f"File '{file_name}' not found. Please make sure the file exists.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"File '{file_name}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error parsing the Excel file '{file_name}'. Please make sure it is a valid Excel file.")
    exit(1)

# Define a function to extract values from JSON
def extract_from_json(json_str, field):
    try:
        json_data = json.loads(json_str)
        return json_data.get(field, '')
    except json.JSONDecodeError:
        return ''

# Extract specific fields from JSON_Serialization column
fields_to_extract = ["DistrictID", "RemoteTagID", "RemoteBatchID", "RemotePEID", "ScanDate", "IsDeleted", "WaybillNo", "SocietyID", "TagNo", "RemoteID", "CreatedBy", "CreatedDate", "UpdatedBy", "UpdatedDate"]
for field in fields_to_extract:
    df[field] = df['JSON_Serialization'].apply(lambda x: extract_from_json(x, field))

# Remove duplicates using tag number
df.drop_duplicates(subset='TagNo', keep='first', inplace=True)


#drop the primary field
df=df.drop(['JSON_Serialization'], axis=1)

# Print the modified DataFrame
print(df)

# Prompt the user for the output Excel file name
output_file_name = input("Enter the output Excel file name (including extension): ")

# Save the modified DataFrame to Excel
df.to_csv(output_file_name, index=False)
print(f"Modified DataFrame (with duplicates removed) saved to '{output_file_name}'.")


# In[ ]:




