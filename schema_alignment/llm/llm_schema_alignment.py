import json
import data.extractor as extraction
from schema_alignment.llm.utils.llm_connector import query_llm


def align_schema_with_llm(df, target_mapping):
    """
    Uses the LLM to align the schema of the provided DataFrame with a target mapping.

    Args:
        df: A pandas DataFrame whose columns need to be aligned.
        target_mapping: A dictionary where keys represent expected source names and values
                        represent the standardized field names.

    Returns:
        A dictionary with the aligned mapping as returned by the LLM or None if an error occurs.
    """
    # Get the list of current columns from the DataFrame
    df_columns = df.columns.tolist()

    # Define the system prompt that instructs the LLM about its role
    system_prompt = (
        "You are a data schema alignment assistant. "
        "Your task is to map the dataset's column names to standardized field names."
    )

    # Construct the user prompt that describes the current schema and the target mapping.
    user_prompt = (
        f"I have a dataset with the following columns: {df_columns}.\n"
        f"I want to map these columns to standardized field names as described in the target mapping below.\n"
        f"Target mapping (source: standardized): {target_mapping}\n\n"
        "Please return a Python dictionary mapping each column name from the dataset to "
        "its aligned standardized field name. If a column cannot be mapped, you can set its value to null."
    )

    # Query the LLM with the prompts
    response = query_llm(system_prompt, user_prompt)

    # Handle and parse the LLM response. Adjust this part if your LLM returns a different format.
    if response is not None and "result" in response:
        try:
            # Expecting that the LLM returns a JSON string that is a valid dictionary
            aligned_mapping = json.loads(response["result"])
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
    df = dataframes[0]

    # (Optional) Remove the 'file_name' column if present
    if 'file_name' in df.columns:
        df = df.drop(columns=['file_name'])

    # Define a target mapping example (customize as per your actual standardized schema)
    target_mapping = {
        'name': 'company_name',
        'address': 'company_address',
        'industry': 'company_industry',
        'city': 'company_city',
        # Add other mappings as required...
    }

    # Call the alignment function
    aligned_schema = align_schema_with_llm(df, target_mapping)

    # Print the aligned mapping
    print("Aligned Schema Mapping:")
    print(aligned_schema)


if __name__ == "__main__":
    test_schema_alignment()
