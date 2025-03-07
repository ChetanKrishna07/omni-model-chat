import autogen
import dotenv

dotenv.load_dotenv()

config_list_agent = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={"gpt-3.5-turbo": "OPENAI_API_KEY"},
)

config_list_agent = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={"gpt-3.5-turbo": "OPENAI_API_KEY"},
)

llm_config = {
    "cache_seed": 44,
    "temperature": 0,
    "config_list": config_list_agent,
    "timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name = "Admin",
    llm_config = llm_config,
    system_message = "A human admin who decides which agent to use based on the agent's capabilities and the user's request. You only use one agent per user request.",
    code_execution_config = {
        "work_dir": "code",
        "use_docker": False,
    },
    human_input_mode="TERMINATE"
)

artist = autogen.AssistantAgent(
    name = "Artist",
    llm_config = llm_config,
    system_message = "You are an agent capable of doing creative tasks that do not need any analysis or reasoning. Examples of your tasks include creative writing, poetry, and drawing.",
)

mathematician = autogen.AssistantAgent(
    name = "Mathematician",
    llm_config = llm_config,
    system_message = "You are an agent capable of solving mathematical problems and providing explanations for the solutions.",
)

coder = autogen.AssistantAgent(
    name = "Coder",
    llm_config = llm_config,
    system_message = "You are an agent capable of writing code in various programming languages.",
)

group_chat = autogen.GroupChat(
    agents = [user_proxy, artist, mathematician, coder],
    messages = [],
    max_round = 10,
)

manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

if __name__ == "__main__":
    input_message = input("> ")
    user_proxy.initiate_chat(
        manager,
        message=input_message,
    )