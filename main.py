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

# Note: This example uses mock tools instead of real APIs for demonstration purposes
def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


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



planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=four_o,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        MathAgent: Expert in doing only math problems
        CreativeAgent: Expert only in creative tasks
        CodingAgent: Expert only in writing code
        GenericAgent: Expert in general tasks, it can also perform research for the data needed to execute the task.

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

generic_agent = AssistantAgent(
    "GenericAgent",
    description="An agent for general tasks. Or general text generation.",
    model_client=four_o,
    system_message="""
    You are a generic agent.
    You can perform general tasks that do not require specialized knowledge. You also have the ability to perform research to gather data needed to execute a task.
    """
)

math_agent = AssistantAgent(
    "MathAgent",
    description="An agent for solving mathematical problems.",
    model_client=o3_mini,
    system_message="""
    You are a math agent.
    Your job is to only solve mathematical problems and provide explanations for the solutions. Present every step in the solution.
    """,
)

creative_agent = AssistantAgent(
    "CreativeAgent",
    description="An agent for creative tasks.",
    model_client=four_o,
    system_message="""
    You are a creative agent.
    Your job is to perform creative tasks that do not require analysis or reasoning.
    """
)

coding_agent = AssistantAgent(
    "CodingAgent",
    description="An agent for writing code.",
    model_client=o3_mini,
    system_message="""
    You are a coding agent.
    Your job is to write code in various programming languages. You are an expert in coding. You provide proper comments and documentation for your code.
    You first give the code enclosed in triple backticks, then the output followed by the explation in plain text and the documentation if required.
    Format:
    ```
    [code]
    ```
    Output: 
    ```
    [output]
    ```
    Explanation:
    [explanation]
    Documentation:
    [documentation]
    """
)

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    If you have not seen the data, ask for it.
    """,
)

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
    [planning_agent, math_agent, creative_agent, coding_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)

task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"

# Use asyncio.run(...) if you are running this in a script.
asyncio.run(Console(team.run_stream(task=task)))