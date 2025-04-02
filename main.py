import dotenv
dotenv.load_dotenv()
from langchain_community.callbacks.manager import get_openai_callback

from chain_config import math_chain, preprocessing_chain
from helpers import extract_filtered_problem, parse_math_solution, calculate_cost
from data_process import get_tasks_gsm_8k


def run_task_solo(task: dict[str, str], verbose=False) -> bool:
    """
    Run the task with the math chain only.
    
    Args:
        task (dict): A dictionary containing the "Question" and "Answer".
    
    Returns:
        bool: Returns True if the answer is correct, False otherwise.
    """
    # Wrap the math chain invocation in a callback context
    with get_openai_callback() as cb:
        math_out = math_chain.invoke({"filtered_problem": task["Question"].strip()})
    
    if verbose:
        print("\nMath Chain Token Details (Solo Task):")
        print("Prompt tokens:", cb.prompt_tokens)
        print("Completion tokens:", cb.completion_tokens)
        print("Total tokens:", cb.total_tokens)
        
    cost = calculate_cost(cb.prompt_tokens, cb.completion_tokens, "o3-mini-2025-01-31")
    
    if verbose:
        print(f"Cost for math chain: ${cost}e-6")
    
    math_text = math_out.content
    solution_dict = parse_math_solution(math_text)
    
    if verbose:
        print("\nSolution:")
        print("Steps:\n", solution_dict["Steps"])
        print("Answer:\n", solution_dict["Answer"])
    
    if "UNSOLVABLE" in solution_dict["Answer"]:
        if verbose:
            print("The problem is unsolvable.")
        return False, cost, solution_dict
    else:
        try:
            if float(solution_dict["Answer"].strip()) == float(task["Answer"].strip()):
                if verbose:
                    print("The answer is correct!")
                return True, cost, solution_dict
            else:
                if verbose:
                    print("The answer is incorrect.")
                return False, cost, solution_dict
        except ValueError:
            if verbose:
                print("The answer format is incorrect or not a number.")
            return False, cost, solution_dict
    
def run_task_omni(task: dict[str, str], verbose=False) -> bool:
    """
    Run the task through the preprocessing and math agent chains.
    
    Args:
        task (dict): A dictionary containing the "Question" and "Answer".
        
    Returns:
        bool: Returns True if the answer is correct, False otherwise.
    """
    # Preprocessing chain invocation with callback
    with get_openai_callback() as cb_pre:
        preprocessed_out = preprocessing_chain.invoke({"problem": task["Question"].strip()})
        
    if verbose:
        print("\nPreprocessing Chain Token Details:")
        print("Prompt tokens:", cb_pre.prompt_tokens)
        print("Completion tokens:", cb_pre.completion_tokens)
        print("Total tokens:", cb_pre.total_tokens)
    cost_pre = calculate_cost(cb_pre.prompt_tokens, cb_pre.completion_tokens, "gpt-4o")
    
    preprocessed_text = preprocessed_out.content
    preprocessed_dict = extract_filtered_problem(preprocessed_text)
    
    if verbose:
        print("\nPre-processed Thinking:\n", preprocessed_dict["thinking"])
        print("\nFiltered Problem:\n", preprocessed_dict["FilteredProblem"])

    # Math chain invocation with callback
    with get_openai_callback() as cb_math:
        math_out = math_chain.invoke({"filtered_problem": preprocessed_dict["FilteredProblem"]})
    
    if verbose:
        print("\nMath Chain Token Details (Omni Task):")
        print("Prompt tokens:", cb_math.prompt_tokens)
        print("Completion tokens:", cb_math.completion_tokens)
        print("Total tokens:", cb_math.total_tokens)
    cost_math = calculate_cost(cb_math.prompt_tokens, cb_math.completion_tokens, "o3-mini-2025-01-31")
    # total_cost = cost_pre + cost_math
    total_cost = cost_math
    
    if verbose:
        print(f"Total Cost for Omni Task: ${total_cost}e-6")
    
    math_text = math_out.content
    solution_dict = parse_math_solution(math_text)
    
    if verbose:
        print("\nSolution:")
        print("Steps:\n", solution_dict["Steps"])
        print("Answer:\n", solution_dict["Answer"])
    
    if "UNSOLVABLE" in solution_dict["Answer"]:
        if verbose:
            print("The problem is unsolvable.")
        return False, total_cost, solution_dict
    else:
        try:
            if float(solution_dict["Answer"].strip()) == float(task["Answer"].strip()):
                if verbose:
                    print("The answer is correct!")
                return True, total_cost, solution_dict
            else:
                if verbose:
                    print("The answer is incorrect.")
                return False, total_cost, solution_dict
        except ValueError:
            if verbose:
                print("The answer format is incorrect or not a number.")
            return False, total_cost, solution_dict

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
        omni_cost += omni_price
        
        if verbose:
            print("--" * 50)
        if correct:
            omni_correct += 1
            omni_correct_cost += omni_price
            
        if verbose:
            print("Running Solo Task:\n")
            
        correct, solo_price, solo_sol = run_task_solo(task, verbose=verbose)
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
    sub_set = tasks[:5]
    
    print("Running Analysis on Tasks...\n")
    
    omni_results = run_analysis(sub_set, verbose=True)
    
    print("Analysis Complete.\n")
    
    print(f"\nSummary of Results:")
    print(f"Omni Task Correct: {omni_results['omni_correct']}/{len(sub_set)}")
    print(f"Average Omni Task Cost: ${omni_results['omni_cost'] / len(sub_set)} / 1M tokens")
    
    print(f"Omni Task Correct Cost: ${omni_results['omni_correct_cost'] / omni_results['omni_correct']} / 1M tokens")
    
    print("--" * 50)
    
    print(f"Solo Task Correct: {omni_results['solo_correct']}/{len(sub_set)}")
    print(f"Average Solo Task Cost: ${omni_results['solo_cost'] / len(sub_set)} / 1M tokens")
    
    print(f"Solo Task Correct Cost: ${omni_results['solo_correct_cost'] / omni_results['solo_correct']} / 1M tokens")

if __name__ == "__main__":
    main()