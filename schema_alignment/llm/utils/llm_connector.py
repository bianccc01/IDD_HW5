import json
import os

import requests
import dotenv

from schema_alignment.llm.utils.llm_adapter import ResponseAdapter


def query_llm(system_prompt, user_prompt):
    """
    Query the LLM with the given system and user prompts.

    Args:
        system_prompt: The system prompt to configure the LLM.
        user_prompt: The user prompt containing the specific task.

    Returns:
        The response from the LLM as a dictionary.
    """
    headers = {'Content-Type': 'application/json'}

    # Get env_config from .env file
    dotenv.load_dotenv()

    env_config = {
        "LLM_ENDPOINT_TYPE": os.getenv("LLM_ENDPOINT_TYPE"),
        "LLM_URL": os.getenv("LLM_URL"),
        "OPENROUTER_KEY": os.getenv("OPENROUTER_KEY"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE")
    }

    if env_config["LLM_ENDPOINT_TYPE"] == 'openrouter':
        url = env_config["LLM_URL"]
        headers["Authorization"] = f"Bearer {env_config['OPENROUTER_KEY']}"

        data = {
            "model": env_config["LLM_MODEL"],
            "temperature": 0.5,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

    elif env_config["LLM_ENDPOINT_TYPE"] == 'ollama':
        url = env_config["LLM_URL"]

        data = {
            "model": env_config["LLM_MODEL"],
            "temperature": env_config["LLM_TEMPERATURE"],
            "messages": [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n{user_prompt}",
                }
            ],
            "stream": False
        }

    else:
        raise ValueError("Invalid LLM_ENDPOINT_TYPE. Please set it to 'openrouter' or 'ollama'.")

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return ResponseAdapter(response.json(), env_config["LLM_ENDPOINT_TYPE"]).to_dict()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
