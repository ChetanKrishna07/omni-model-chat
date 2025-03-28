from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import dotenv
import os

dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=openai_api_key
    )

four_o = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=openai_api_key
    )

o3_mini = OpenAIChatCompletionClient(
    model="o3-mini-2025-01-31",
    api_key=openai_api_key
    )


pre_processing_agent = AssistantAgent(
    "PreProcessingAgent",
    description="Removes extraneous information from a given math word problem.",
    model_client=four_o,
    system_message="""
    You are a pre-processing agent.
    Your job is to remove any extraneous information from the math word problem that is unnecessary for solving the problem.
    If a given piece of information is irrelevant to the problem, you should remove it.
    If a given piece of information has a slighest relevance to the problem, you should keep it.
    Only remove information that is irrelevant to what the problem is asking.
    
    Explain why you removed the information in the math word problem.
    
    Your response should be in the following format:
    <Thinking>
    [Every step of your thought process to remove the extraneous information]
    </Thinking> 
    <FilteredProblem>
    [Math word problem with extraneous information removed]
    </FilteredProblem>
    """
)

math_agent = AssistantAgent(
    "MathAgent",
    description="An agent for solving mathematical problems.",
    model_client=o3_mini,
    system_message="""
    You are a math agent.
    
    Your job is to only solve mathematical problems and provide explanations for the solutions.
    
    If the information in the math word problem given is not sufficient to solve the problem, respond exactly with <Solution>UNSOLVABLE</Solution> nothing more, nothing less.
    If the problem if unsolvable, respond exactly with <Solution>UNSOLVABLE</Solution> nothing more, nothing less.
    
    If the problem is solvable, present in the following format:
    
    <Solution>
    [Detailed Step-by-Step Solution]
        <Answer>
            [Single Numerical Value]
        </Answer>
    </Solution>
    """
)
    

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=four_o,
    system_message="""
    You are a planning agent.
    Your job to orchastrate the team of agents to solve the math word problem.
    
    Your team members are:
        PreProcessingAgent: Expert in removing extraneous information from math word problems.
        MathAgent: Expert in solving mathematical problems.
    
    You fist give the math word problem to the PreProcessingAgent to remove any extraneous information.
    Then you pass whatever is in the <FilteredProblem> from the PreProcessingAgent's response to the MathAgent to solve the math word problem.
    You finally display the solution to the math word problem exactly as it is given by the MathAgent.
    
    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)


# web_search_agent = AssistantAgent(
#     "WebSearchAgent",
#     description="An agent for searching information on the web.",
#     tools=[search_web_tool],
#     model_client=model_client,
#     system_message="""
#     You are a web search agent.
#     Your only tool is search_tool - use it to find information.
#     You make only one search call at a time.
#     Once you have the results, you never do calculations based on them.
#     """,
# )

# data_analyst_agent = AssistantAgent(
#     "DataAnalystAgent",
#     description="An agent for performing calculations.",
#     model_client=model_client,
#     tools=[percentage_change_tool],
#     system_message="""
#     You are a data analyst.
#     Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
#     If you have not seen the data, ask for it.
#     """,
# )

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

team = SelectorGroupChat(
    [planning_agent, math_agent, pre_processing_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
)

task = """

Problem:

Samantha, an avid birdwatcher, wakes up at 6:45 AM every Saturday to prepare for her weekend birdwatching trip. She always packs a thermos of coffee (450 ml), three granola bars, a notebook with 120 pages (only 15 are used), and a set of binoculars with 10x magnification. On a particular Saturday, she drives 35 miles to the Silver Creek Nature Reserve, where she plans to spend exactly 4 hours observing birds.

There are 12 types of birds commonly seen at the reserve, but she's especially interested in observing the red-tailed hawk. Last week, she saw 4 of them in 3 hours, but only 2 of them were mature adults. The reserve is open from 8:00 AM to 6:00 PM, and parking costs $3 per hour.

This Saturday, Samantha arrived at 8:15 AM and parked her car. She spent 45 minutes walking to the northern overlook and 30 minutes setting up her observation spot. While birdwatching, she recorded that the average number of red-tailed hawks spotted per hour increased by 0.5 compared to last week. Afterward, she stopped by the reserve's gift shop, where she spent $18.50 on souvenirs and bought a book on migratory birds discounted by 25% off its original $24 price.

Question:
How many red-tailed hawks did Samantha observe this Saturday?

"""

# Use asyncio.run(...) if you are running this in a script.
asyncio.run(Console(team.run_stream(task=task)))