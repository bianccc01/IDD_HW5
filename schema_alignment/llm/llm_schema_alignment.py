import json
import re

import pandas as pd
import data.extractor as extraction
from schema_alignment import mediated_schema
from utils.llm_connector import query_llm


def align_schema_with_llm(df, target_mapping):
    """
    Uses the LLM to align the schema of the provided DataFrame with a target mapping.

    Args:
        df: A pandas DataFrame whose columns need to be aligned.
        target_mapping: A dictionary where keys represent expected source names and values
                        represent the standardized field names.
    """

    # Define the system prompt that instructs the LLM about its role
    system_prompt = (
        "You are a data schema alignment REST server. "
        "Your task is to map the dataset's column names to standardized field names."
        "you should align the schema of the provided DataFrame with the target mapping."
        "you must return a valid json like this: {'mapped_schema': 'source_column': 'target_column'}"
    )

    # Construct the user prompt that describes the current schema and the target mapping.
    user_prompt = f"""
            {{
                "current_schema": {{
                    "columns": {df.columns.tolist()},
                    "data_sample": {df.head().to_dict(orient='records')}
                }},
                "target_mapping": {{
                    "columns": {mediated_schema},
                    "data_sample": {target_mapping.head(3).to_dict(orient='records')}
                }}
            }}
    """

    # Query the LLM with the prompts
    response = query_llm(system_prompt, user_prompt)

    # Handle and parse the LLM response. Adjust this part if your LLM returns a different format.
    if response is not None and response.get("message") is not None:
        try:
            # Expecting that the LLM returns a JSON string that is a valid dictionary
            # clean everything that is not a valid json and is not between curly brackets
            message = response.get("message", "")
            match = re.search(r'\{.*\}', message, re.DOTALL)

            if match:
                clean_response = match.group(0)  # Estrai il JSON
                try:
                    aligned_mapping = json.loads(clean_response)  # Verifica se è un JSON valido
                except json.JSONDecodeError:
                    aligned_mapping = None
            else:
                aligned_mapping = None  # Nessun JSON trovato

            return aligned_mapping
        except Exception as e:
            print("Error parsing LLM response:", e)
            return None
    else:
        print("No valid response from LLM.")
        return None


def test_schema_alignment():
    """
    Extracts a test DataFrame, applies the schema alignment using LLM, and prints the result.
    """
    # Extract test data (adjust the path as needed)
    dataframes = extraction.extract_data('../../data/schema_alignment/test')

    if not dataframes:
        print("No valid DataFrame found for schema alignment.")
        return

    # For demonstration, pick the first DataFrame from the list
    df = dataframes[1]

    # (Optional) Remove the 'file_name' column if present
    if 'file_name' in df.columns:
        df = df.drop(columns=['file_name'])

    # Define a target mapping example (customize as per your actual standardized schema)
    target_mapping = pd.read_csv('../../data/schema_alignment/created/llm/example.csv')

    # Call the alignment function
    aligned_schema = align_schema_with_llm(df, target_mapping)

    # Print the aligned mapping
    print("Aligned Schema Mapping:")
    print(aligned_schema)


def align_schemas():
    # Extract test data (adjust the path as needed)
    dataframes = extraction.extract_data('../../data/schema_alignment/test')

    if not dataframes:
        print("No valid DataFrame found for schema alignment.")
        return

    aligned_schemas = []

    for df in dataframes:
        # (Optional) Remove the 'file_name' column if present
        if 'file_name' in df.columns:
            df = df.drop(columns=['file_name'])

        # Define a target mapping example (customize as per your actual standardized schema)
        target_mapping = pd.read_csv('../../data/schema_alignment/created/llm/example.csv')

        # Call the alignment function
        print("Aligning schema for DataFrame:", df['file_name'][0])
        aligned_schema = align_schema_with_llm(df, target_mapping)

        aligned_schemas.append(aligned_schema)

    return aligned_schemas

