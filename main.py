import re
import os
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
import dotenv
dotenv.load_dotenv()

from langchain_community.callbacks.manager import get_openai_callback

# Define pricing rates (dollars per 1M tokens)
pricing = {
    "o3-mini-2025-01-31": {"prompt": 0.150, "completion": 0.600 },
    "gpt-4o-mini": {"prompt": 0.005, "completion": 0.01}
}

openai_api_key = os.getenv("OPENAI_API_KEY")

llm_gpt4o = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)
llm_o3mini = ChatOpenAI(model="o3-mini-2025-01-31", openai_api_key=openai_api_key, temperature=None)


preprocessing_prompt_template = """
You are a pre-processing agent.
Your job is to remove any extraneous information from the math word problem that is unnecessary for solving the problem.
If a given piece of information is completely irrelevant to the problem, you should remove it.
If a given piece of information has a slighest relevance to the problem, you should keep it.
If you are not 100% sure about the relevance of a piece of information, you should keep it.
Only remove information that is completely irrelevant to what the problem is asking.

Explain why you removed the information in the math word problem.

Your response should be in the following format:
<Thinking>
[Every step of your thought process to remove the extraneous information]
</Thinking> 
<FilteredProblem>
[Math word problem with extraneous information removed]
</FilteredProblem>

Math Question: {problem}
"""

preprocessing_prompt = PromptTemplate(
    input_variables=["problem"],
    template=preprocessing_prompt_template
)

# The preprocessing chain uses the GPT-4 model
preprocessing_chain = preprocessing_prompt | llm_gpt4o


math_prompt_template = """
You are a math agent.
Your job is to only solve mathematical problems and provide explanations for the solutions.

If the information in the math word problem given is not sufficient to solve the problem, respond exactly with UNSOLVABLE in the <Answer> tag nothing more, nothing less.
If the problem is unsolvable, respond exactly with UNSOLVABLE in the <Answer> tag nothing more, nothing less.
Attempt to solve the problem using the information provided in the math word problem, and if you cannot find a solution, resopond with UNSOLVABLE in the <Answer> tag.
Always simplify your answer to the simplest form possible, and relevant to the problem.

If the problem is solvable, present the solution in the following format:

<Solution>
    <Steps>
        [Detailed Step-by-Step Solution]
    </Steps>
    <Answer>
        [Single Numerical Value] / UNSOLVABLE
    </Answer>
</Solution>

Math Question: {filtered_problem}
"""

math_prompt = PromptTemplate(
    input_variables=["filtered_problem"],
    template=math_prompt_template
)

# The math chain uses the o3-mini model
math_chain = math_prompt | llm_o3mini


def extract_filtered_problem(text: str) -> dict:
    """Extracts the content within <FilteredProblem> tags."""
    filter_problem = re.search(r"<FilteredProblem>(.*?)</FilteredProblem>", text, re.DOTALL)
    thinking = re.search(r"<Thinking>(.*?)</Thinking>", text, re.DOTALL)
    if filter_problem:
        return {
            "thinking": thinking.group(1).strip() if thinking else "",
            "FilteredProblem": filter_problem.group(1).strip()
        }
    else:
        raise ValueError("FilteredProblem not found in the pre-processing output.")


def parse_math_solution(text: str) -> dict:
    """Parses the math agent's XML output to extract steps and answer."""
    steps_match = re.search(r"<Steps>(.*?)</Steps>", text, re.DOTALL)
    answer_match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if steps_match and answer_match:
        return {
            "Steps": steps_match.group(1).strip(),
            "Answer": answer_match.group(1).strip()
        }
    elif "UNSOLVABLE" in text:
        return {"Steps": "", "Answer": "UNSOLVABLE"}
    else:
        raise ValueError("Solution format not recognized.")


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost based on token counts and pricing rates."""
    rates = pricing.get(model)
    if rates:
        cost = (prompt_tokens) * rates["prompt"] + (completion_tokens) * rates["completion"]
        return cost
    else:
        return 0.0


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
        print(f"Cost for math chain: ${cost:.4f}e-6")
    
    math_text = math_out.content
    solution_dict = parse_math_solution(math_text)
    
    if verbose:
        print("\nSolution:")
        print("Steps:\n", solution_dict["Steps"])
        print("Answer:\n", solution_dict["Answer"])
    
    if "UNSOLVABLE" in solution_dict["Answer"]:
        if verbose:
            print("The problem is unsolvable.")
        return False, cost
    else:
        try:
            if float(solution_dict["Answer"].strip()) == float(task["Answer"].strip()):
                if verbose:
                    print("The answer is correct!")
                return True, cost
            else:
                if verbose:
                    print("The answer is incorrect.")
                return False, cost
        except ValueError:
            if verbose:
                print("The answer format is incorrect or not a number.")
            return False, cost
    

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
    total_cost = cost_pre + cost_math
    
    if verbose:
        print(f"Total Cost for Omni Task: ${total_cost:.4f}e-6")
    
    math_text = math_out.content
    solution_dict = parse_math_solution(math_text)
    
    if verbose:
        print("\nSolution:")
        print("Steps:\n", solution_dict["Steps"])
        print("Answer:\n", solution_dict["Answer"])
    
    if "UNSOLVABLE" in solution_dict["Answer"]:
        if verbose:
            print("The problem is unsolvable.")
        return False, total_cost
    else:
        try:
            if float(solution_dict["Answer"].strip()) == float(task["Answer"].strip()):
                if verbose:
                    print("The answer is correct!")
                return True, total_cost
            else:
                if verbose:
                    print("The answer is incorrect.")
                return False, total_cost
        except ValueError:
            if verbose:
                print("The answer format is incorrect or not a number.")
            return False, total_cost


def run_analysis(tasks, verbose=False):
    
    omni_correct = 0
    omni_cost = 0
    solo_correct = 0
    solo_cost = 0
    
    for task in tasks:
        if verbose:
            print("--" * 50)
            print(f"Running Task: {task['Question'].strip()}")
            print("--" * 50)
            print("Running Omni Task:\n")
        correct, omni_price = run_task_omni(task, verbose=verbose)
        omni_cost += omni_price
        
        if verbose:
            print("--" * 50)
        if correct:
            omni_correct += 1
            
        if verbose:
            print("Running Solo Task:\n")
            
        correct, solo_price = run_task_solo(task, verbose=verbose)
        solo_cost += solo_price
        
        if verbose:
            print("--" * 50)
        if correct:
            solo_correct += 1
            
        if verbose:
            print("--" * 50)
    
    return {
        "omni_correct": omni_correct,
        "omni_cost": omni_cost,
        "solo_correct": solo_correct,
        "solo_cost": solo_cost,
    }


def main():
    task_1 = """
    Problem:

    Samantha, an avid birdwatcher, wakes up at 6:45 AM every Saturday to prepare for her weekend birdwatching trip. She always packs a thermos of coffee (450 ml), three granola bars, a notebook with 120 pages (only 15 are used), and a set of binoculars with 10x magnification. On a particular Saturday, she drives 35 miles to the Silver Creek Nature Reserve, where she plans to spend exactly 4 hours observing birds.

    There are 12 types of birds commonly seen at the reserve, but she's especially interested in observing the red-tailed hawk. Last week, she saw 4 of them in 3 hours, but only 2 of them were mature adults. The reserve is open from 8:00 AM to 6:00 PM, and parking costs $3 per hour.

    This Saturday, Samantha arrived at 8:15 AM and parked her car. She spent 45 minutes walking to the northern overlook and 30 minutes setting up her observation spot. While birdwatching, she recorded that the average number of red-tailed hawks spotted per hour increased by 0.5 compared to last week. Afterward, she stopped by the reserve's gift shop, where she spent $18.50 on souvenirs and bought a book on migratory birds discounted by 25% off its original $24 price.

    Question:
    How many red-tailed hawks did Samantha observe this Saturday? (Present the final answer as a single numerical value, not fractionals)
    """

    task_2 = """
    One number is 11 more than another number. Find the two numbers if three
    times the larger exceeds four times the smaller number by 4.
    """
    
    task = """
    Manuel opened a savings account with an initial deposit of 177 dollars. He can withdraw from the account every 2 months. If he wants to have 500 dollars after the end of next 19 weeks, how much must he save each week?
    """
    
    tasks = [{
        "Question": task_1,
        "Answer": "7"
    }, {
        "Question": task,
        "Answer": "17.0"
    }]
    
    print("Running Analysis on Tasks...\n")
    
    omni_results = run_analysis(tasks, verbose=False)
    
    print("Analysis Complete.\n")
    
    print(f"\nSummary of Results:")
    print(f"Omni Task Correct: {omni_results['omni_correct']}/{len(tasks)}")
    print(f"Average Omni Task Cost: ${omni_results['omni_cost'] / len(tasks):.4f} / 1M tokens")
    print(f"Solo Task Correct: {omni_results['solo_correct']}/{len(tasks)}")
    print(f"Average Solo Task Cost: ${omni_results['solo_cost'] / len(tasks):.4f} / 1M tokens")

if __name__ == "__main__":
    main()