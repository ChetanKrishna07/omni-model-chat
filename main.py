import dotenv
dotenv.load_dotenv()
from data_process import get_tasks_gsm_8k, get_tasks_asdiv, filter_tasks
from completion import run_task_omni, run_task_solo

def run_analysis(tasks, verbose=False):
    
    omni_correct = 0
    omni_cost = 0
    omni_correct_cost = 0
    solo_correct = 0
    solo_cost = 0
    solo_correct_cost = 0
    
    for task in tasks:
        if verbose:
            print("--" * 50)
            print(f"Running Task: {task['Question'].strip()}")
            print(f"Correct Answer: {task['Answer'].strip()}")
            print("--" * 50)
            print("Running Omni Task:\n")
        correct, omni_price, omni_sol = run_task_omni(task, verbose=verbose)
        print(f"Omni Task Cost: ${omni_price}")
        omni_cost += omni_price
        
        if verbose:
            print("--" * 50)
        if correct:
            omni_correct += 1
            omni_correct_cost += omni_price
            
        if verbose:
            print("Running Solo Task:\n")
            
        correct, solo_price, solo_sol = run_task_solo(task, verbose=verbose)
        print(f"Solo Task Cost: ${solo_price}")
        solo_cost += solo_price
        
        if verbose:
            print("--" * 50)
        if correct:
            solo_correct += 1
            solo_correct_cost += solo_price
            
        if verbose:
            print("--" * 50)
            
    
    return {
        "omni_correct": omni_correct,
        "omni_cost": omni_cost,
        "omni_correct_cost": omni_correct_cost,
        "solo_correct": solo_correct,
        "solo_cost": solo_cost,
        "solo_correct_cost": solo_correct_cost
    }


def main():
    # task_1 = """
    # Problem:

    # Samantha, an avid birdwatcher, wakes up at 6:45 AM every Saturday to prepare for her weekend birdwatching trip. She always packs a thermos of coffee (450 ml), three granola bars, a notebook with 120 pages (only 15 are used), and a set of binoculars with 10x magnification. On a particular Saturday, she drives 35 miles to the Silver Creek Nature Reserve, where she plans to spend exactly 4 hours observing birds.

    # There are 12 types of birds commonly seen at the reserve, but she's especially interested in observing the red-tailed hawk. Last week, she saw 4 of them in 3 hours, but only 2 of them were mature adults. The reserve is open from 8:00 AM to 6:00 PM, and parking costs $3 per hour.

    # This Saturday, Samantha arrived at 8:15 AM and parked her car. She spent 45 minutes walking to the northern overlook and 30 minutes setting up her observation spot. While birdwatching, she recorded that the average number of red-tailed hawks spotted per hour increased by 0.5 compared to last week. Afterward, she stopped by the reserve's gift shop, where she spent $18.50 on souvenirs and bought a book on migratory birds discounted by 25% off its original $24 price.

    # Question:
    # How many red-tailed hawks did Samantha observe this Saturday? (Present the final answer as a single numerical value, not fractionals)
    # """

    # task_2 = """
    # One number is 11 more than another number. Find the two numbers if three
    # times the larger exceeds four times the smaller number by 4.
    # """
    
    # task = """
    # Manuel opened a savings account with an initial deposit of 177 dollars. He can withdraw from the account every 2 months. If he wants to have 500 dollars after the end of next 19 weeks, how much must he save each week?
    # """
    
    # tasks = [{
    #     "Question": task_1,
    #     "Answer": "7"
    # }, {
    #     "Question": task,
    #     "Answer": "17.0"
    # }]
    
    tasks = get_tasks_gsm_8k()
    # tasks = get_tasks_asdiv()
    long_tasks = filter_tasks(tasks, 54, 500)
    sub_set = long_tasks[:5]
    
    print("Running Analysis on Tasks...\n")
    
    omni_results = run_analysis(sub_set, verbose=True)
    
    print("Analysis Complete.\n")
    
    print(f"\nSummary of Results:")
    print(f"Omni Task Correct: {omni_results['omni_correct']}/{len(sub_set)}")
    print(f"Average Omni Task Cost: ${omni_results['omni_cost'] / len(sub_set)} / 1 problem")
    
    print(f"Omni Task Correct Cost: ${omni_results['omni_correct_cost'] / omni_results['omni_correct']} / 1 problem")
    
    print("--" * 50)
    
    print(f"Solo Task Correct: {omni_results['solo_correct']}/{len(sub_set)}")
    print(f"Average Solo Task Cost: ${omni_results['solo_cost'] / len(sub_set)} / 1 problem")
    
    print(f"Solo Task Correct Cost: ${omni_results['solo_correct_cost'] / omni_results['solo_correct']} / 1 problem")

if __name__ == "__main__":
    main()