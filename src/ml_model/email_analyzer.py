import json

import pandas as pd
from huggingface_hub import InferenceClient

from src.database import settings
from src.statistics.load import load_data_email


def huggingface_llama(email_filename):
    if not settings.HUGGINGFACE_TOKEN:
        raise ValueError(
            "HUGGINGFACE_TOKEN is not set. Add it to .env in the project root "
            "(see .env.example) and install python-dotenv, or export HUGGINGFACE_TOKEN."
        )

    df_email = load_data_email(email_filename)
    email_text = "\n".join(df_email.iloc[:, 0].astype(str).tolist())

    client = InferenceClient(
        model="meta-llama/Llama-3.3-70B-Instruct",
        token=settings.HUGGINGFACE_TOKEN
    )

    prompt = f"""
    You are a data extraction assistant. Your task is to extract key information from the following email and return it strictly as a single JSON object.

    [Email Content]:
    {email_text}

    Extract the following 15 fields:
    1. 'education_level' ('Master', 'Bachelor', 'High School', 'PhD', )
    2. 'industry_name' (string)
    3. 'job_role_name' (string)
    4. 'company_size' (Medium', 'Small', 'Large')
    5. 'job_level' ('Entry', 'Senior', 'Mid')
    6. 'ai_adoption_level' (string)
    7. 'age' (float)
    8. 'years_of_experience' (float)
    9. 'routine_task_percentage' (float)
    10. 'creativity_requirement' (float:score 0- 100)
    11. 'human_interaction_level' (float:score 0- 10)
    12. 'number_of_ai_tools_used' (float:score 0- 10)
    13. 'ai_usage_hours_per_week' (float:score 0- 100)
    14. 'tasks_automated_percentage' (float:score 0- 100)
    15. 'ai_training_hours' (float:score 0- 100)

    Strict Constraints:
    1. Output MUST be a single, valid JSON object and nothing else. No markdown wrappers (like ```dict), no conversational filler, no explanations.
    2. Do not use Null/None. If a value is missing:
    - For string fields, use an empty string "".
    - For float fields, use -1.0.
    3. Ensure all string field values are normalized into English.
    4. Strictly follow the data types specified above.
    5. If a value is empty, you need to analyze it and then fill it in.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a Email-Data snalysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.1
    )

    ai_return_content = chat_completion.choices[0].message.content
    data_dict = json.loads(ai_return_content)
    return pd.DataFrame([data_dict])
