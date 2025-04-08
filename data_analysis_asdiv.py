import pandas as pd
from data_process import get_tasks_asdiv
from completion import run_task_solo, run_task_omni

def get_price(row):
    solo_correct, solo_cost, solo_sol = run_task_solo({"Question": row["Question"], "Answer": row["Answer"]}, verbose=True)
    omni_correct, omni_cost, omni_sol = run_task_omni({"Question": row["Question"], "Answer": row["Answer"]}, verbose=True)
    
    return solo_cost, omni_cost, solo_correct, omni_correct


print("Loading tasks...")
asdiv_tasks = get_tasks_asdiv()

print("Converting tasks to DataFrame...")
asdiv_df = pd.DataFrame(asdiv_tasks)

print("Getting the number of words in each question...")
asdiv_df['no_of_words'] = asdiv_df['Question'].apply(lambda x: len(x.split()))

print("Calculating costs and correctness...")
for index, row in asdiv_df.iterrows():
    solo_cost, omni_cost, solo_correct, omni_correct = get_price(row)
    asdiv_df.at[index, 'solo_cost'] = solo_cost
    asdiv_df.at[index, 'omni_cost'] = omni_cost
    asdiv_df.at[index, 'solo_correct'] = solo_correct
    asdiv_df.at[index, 'omni_correct'] = omni_correct
    if index % 100 == 0:
        asdiv_df.to_csv(f'Data Analysis/asdiv_analysis_{index}.csv', index=False)
        print(f"Processed {index} rows...")

print("Saving results to CSV...")
asdiv_df.to_csv('Data Analysis/asdiv_analysis.csv', index=False)  # Save the dataframe to a CSV file

