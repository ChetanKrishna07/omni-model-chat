import os
import json
import re
import pandas as pd
import xml.etree.ElementTree as ET

def get_tasks_asdiv():
    xml_data = open('ASDiv.xml', 'r').read()
    root = ET.fromstring(xml_data)

    data = []

    # Loop over each 'Problem' element
    for problem in root.findall('.//Problem'):
        problem_data = {
            'ID': problem.get('ID'),
            'Grade': problem.get('Grade'),
            'Source': problem.get('Source'),
            'Body': problem.find('Body').text if problem.find('Body') is not None else '',
            'Question': problem.find('Question').text if problem.find('Question') is not None else '',
            'Solution-Type': problem.find('Solution-Type').text if problem.find('Solution-Type') is not None else '',
            'Answer': problem.find('Answer').text if problem.find('Answer') is not None else '',
            'Formula': problem.find('Formula').text if problem.find('Formula') is not None else ''
        }
        data.append(problem_data)

    # Create a dataFrame
    df = pd.DataFrame(data)
    df['FullQuestion'] = df['Body'] + ' ' + df['Question']
    df['Answer'] = df['Answer'].str.replace(r'\s*\(.*\)', '', regex=True)
    df = df.drop(['Source','Grade','Solution-Type','ID','Formula'], axis=1)


    tasks = []

    for row in df.iterrows():
        data = row[1]
        tasks.append({
            "Question": data['FullQuestion'],
            "Answer": data['Answer']
        })


    return tasks


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