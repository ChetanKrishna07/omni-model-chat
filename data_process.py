import os
import json
import re
import pandas as pd

def get_tasks_gsm_8k(hint=None):
    """
    Load tasks from a JSON file in the specified directory.
    
    Args:
        data_dir (str): The directory containing the JSON file with tasks.
        
    Returns:
        list: A list of tasks loaded from the JSON file.
    """
    
    splits = {'train': 'socratic/train-00000-of-00001.parquet', 'test': 'socratic/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    
    tasks = []
    
    for row in df.iterrows():
        data = row[1]
        
        problem = data.question.strip()
        answer = data.answer.strip()
        
        if re.search(r"#### \d*", answer):
            # print(answer) 
            answer_extract = re.search(r"#### (\d*)", answer)
            if answer_extract:
                answer_extract = answer_extract.group(1)
                dec = answer_extract.split('.')
                if len(dec) > 1:
                    if dec[0] == '':
                        answer_extract = '0.' + dec[1]
            # print(answer_extract)
            print('pg is kewl')
        else:
            print(answer)
            print('pg is not kewl')
        
        tasks.append({
            "Question": problem,
            "Answer": answer_extract
        })
                
        if hint == 'adv':
            question = f"{problem} \nHINT: {data['adv_hint'].strip()}"
        elif hint == 'normal':
            question = f"{problem} \nHINT: {data['hint'].strip()}"
        else:
            question = problem
    
    return tasks

get_tasks_gsm_8k()