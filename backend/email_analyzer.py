import pandas as pd
import load
from config import settings
from huggingface_hub import InferenceClient
import json
import os

'''
def get_email_analysis_data(email_filename):
    """
    Unified interface exposed to other modules.
    Within the same run: First call requests the API and saves to cache; subsequent calls read from cache.
    Upon program exit: Automatically deletes the temporary cache file.
    """
    cache_filename = f"cache_{os.path.splitext(os.path.basename(email_filename))[0]}.json"

    # 1. If cache exists, read from it directly
    if os.path.exists(cache_filename):
        print(f"🔄 [Cache Hit] Reading from temporary cache for this run: {cache_filename}...")
        with open(cache_filename, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        return pd.DataFrame([data_dict])
    
    # 2. If cache does not exist, call the LLM API
    print(f"🔍 [Cache Miss] Preparing for the initial LLM API call...")
    df_result = huggingface_llama(email_filename, cache_filename)

    # 3. Core change: Register an exit hook
    # Instruct Python: "When the process terminates, automatically execute cleanup_cache(cache_filename)"
    atexit.register(cleanup_cache, cache_filename)
    print(f"📌 Registered auto-cleanup hook. {cache_filename} will be deleted upon exit.")

    return df_result


def cleanup_cache(filename):
    """
    Cleanup function: Responsible for removing the temporary cache file.
    """
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"\n🧹 [Auto-Cleanup] Process terminated. Temporary cache {filename} successfully deleted!")
        except Exception as e:
            print(f"\n❌ [Auto-Cleanup] Failed to delete cache file: {e}")
'''

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
    # f"Could you please summarize this email for me in about 50 words?\n\n[Email Content]:\n{email_text}"
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
    AI_Content_df = pd.DataFrame([data_dict])
    return AI_Content_df








