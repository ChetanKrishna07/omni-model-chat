import dotenv
dotenv.load_dotenv()
from langchain_community.callbacks.manager import get_openai_callback
from chain_config import math_chain, preprocessing_chain
from helpers import extract_filtered_problem, parse_math_solution, calculate_cost


def run_task_solo(task: dict[str, str], verbose=False) -> bool:
    """
    Run the task with the math chain only.
    
    Args:
        task (dict): A dictionary containing the "Question" and "Answer".
    
    Returns:
        bool: Return if the answer is correct or not, cost of the task, and the solution dictionary.
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
        bool:  Return if the answer is correct or not, cost of the task, and the solution dictionary.
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
    total_cost = cost_pre + cost_math
    # total_cost = cost_math
    
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
