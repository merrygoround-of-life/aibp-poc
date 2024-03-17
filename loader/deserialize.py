from executor.core.chain_agent import *
from executor.core.embedding_agent import *
from executor.custom.animal_info_router_agent import *
from executor.custom.asr_to_prompt_agent import *
from executor.custom.cat_facts_api_agent import *
from executor.custom.current_location_agent import *
from executor.custom.current_time_agent import *
from executor.custom.dog_facts_api_agent import *
from executor.custom.get_user_profile_agent import *
from executor.custom.response_to_asr_agent import *
from executor.custom.retrival_agent import *
from executor.custom.router_agent import *
from executor.custom.vector_create_agent import *
from executor.custom.vector_query_agent import *


def from_chain_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    chain_agents = []
    for sub_agent_dict in json_dict["chain_agents"]:
        chain_agents.append(from_dict(sub_agent_dict))

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), uuid=json_dict["uuid"], chain_agents=chain_agents)


def from_tool_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    tool_agents = []
    for sub_agent_dict in json_dict["tool_agents"]:
        tool_agents.append(from_dict(sub_agent_dict))

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), uuid=json_dict["uuid"], tool_agents=tool_agents)


def from_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    if "chain_agents" in json_dict:
        return from_chain_dict(json_dict)
    elif "tool_agents" in json_dict:
        return from_tool_dict(json_dict)

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), uuid=json_dict["uuid"])
