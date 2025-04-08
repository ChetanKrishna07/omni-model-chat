import pandas as pd
from data_process import get_tasks_gsm_8k
from completion import run_task_solo, run_task_omni

def get_price(row):
    solo_correct, solo_cost, solo_sol = run_task_solo({"Question": row["Question"], "Answer": row["Answer"]}, verbose=True)
    omni_correct, omni_cost, omni_sol = run_task_omni({"Question": row["Question"], "Answer": row["Answer"]}, verbose=True)
    
    return solo_cost, omni_cost, solo_correct, omni_correct


print("Loading tasks...")
gsm_8k_tasks = get_tasks_gsm_8k()

print("Converting tasks to DataFrame...")
gsm_8k_df = pd.DataFrame(gsm_8k_tasks)

print("Getting the number of words in each question...")
gsm_8k_df['no_of_words'] = gsm_8k_df['Question'].apply(lambda x: len(x.split()))

print("Calculating costs and correctness...")
for index, row in gsm_8k_df.iterrows():
    solo_cost, omni_cost, solo_correct, omni_correct = get_price(row)
    gsm_8k_df.at[index, 'solo_cost'] = solo_cost
    gsm_8k_df.at[index, 'omni_cost'] = omni_cost
    gsm_8k_df.at[index, 'solo_correct'] = solo_correct
    gsm_8k_df.at[index, 'omni_correct'] = omni_correct
    if index % 100 == 0:
        gsm_8k_df.to_csv(f'Data Analysis/gsm_8k_analysis_{index}.csv', index=False)
        print(f"Processed {index} rows...")

print("Saving results to CSV...")
gsm_8k_df.to_csv('Data Analysis/gsm_8k_analysis.csv', index=False)  # Save the dataframe to a CSV file

