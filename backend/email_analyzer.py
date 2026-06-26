import pandas as pd
import load
from config import settings
from huggingface_hub import InferenceClient
import json


def huggingface_llama(email_filename):
    df_email = load.load_data_email(email_filename)
    # print(df_email)

    # iloc[row_index, col_index]
    email_text = "\n".join(df_email.iloc[:, 0].astype(str).tolist())
    token_email = settings.HUGGINGFACE_TOKEN

    # choose your model:
    client = InferenceClient(
        model="meta-llama/Llama-3.3-70B-Instruct",  
        token=token_email
    )
   
    prompt = f"""
    You are a data extraction assistant. Your task is to extract key information from the following email and return it strictly as a single JSON object.

    [Email Content]:
    {email_text}

    Extract the following 10 fields:
    1. 'education_level' ('Master', 'Bachelor', 'High School', 'PhD', )
    2. 'industry_name' (string)
    3. 'job_role_name' (string)
    4. 'job_level' ('Entry', 'Senior', 'Mid')
    5. 'ai_adoption_level' (string)
    6. 'routine_task_percentage' (float)
    7. 'creativity_requirement' (float:score 0- 100) 
    8. 'number_of_ai_tools_used' (float:score 0- 10)
    9. 'ai_usage_hours_per_week' (float:score 0- 100)
    10. 'ai_training_hours' (float:score 0- 100)

    Strict Constraints:
    1. Output MUST be a single, valid JSON object and nothing else. No markdown wrappers (like ```dict), no conversational filler, no explanations.
    2. Do not use Null/None. If a value is missing:
    - For string fields, use an empty string "".
    - For float fields, use -1.0.
    3. Ensure all string field values are normalized into English.
    4. Strictly follow the data types specified above.
    5. If a value is empty, you need to analyze it and then fill it in.
    """

    # Chat-Window API
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a Email-Data snalysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,

        # 0 ~ 2, 0~ 0.3(Rigorous) Temperature controls the balance between randomness (creativity) and predictability (accuracy) in the model's responses.
        temperature=0.1 )

  
    AI_return_Content = chat_completion.choices[0].message.content
    data_dict = json.loads(AI_return_Content)

    print(data_dict)
    AI_Content_df = pd.DataFrame([data_dict])
    return AI_Content_df


