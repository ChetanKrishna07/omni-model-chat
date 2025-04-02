import re

# Define pricing rates (dollars per 1M tokens)
pricing = {
    "o3-mini-2025-01-31": {"prompt": 1.10, "completion": 4.40 },
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600}
}

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
        cost = (prompt_tokens) * rates["prompt"]*(10**-6) + (completion_tokens) * rates["completion"]*(10**-6)
        return cost
    else:
        return 0.0
