import os
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
import dotenv
dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

pre_processing_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)
# math_model = ChatOpenAI(model="o3-mini-2025-01-31", openai_api_key=openai_api_key, temperature=None)
math_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0) 


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
preprocessing_chain = preprocessing_prompt | pre_processing_model


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
math_chain = math_prompt | math_model
