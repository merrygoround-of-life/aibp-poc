from executor.core.chain_agent import *
from executor.core.functioncall_agent import *
from executor.custom.cat_facts_agent import *
from executor.custom.get_current_time_agent import *
from executor.custom.get_user_profile_agent import *
from executor.custom.get_location_agent import *


def from_chain_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    chain_agents = []
    for sub_agent_dict in json_dict["chain_agents"]:
        chain_agents.append(from_dict(sub_agent_dict))

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), name=json_dict["name"], chain_agents=chain_agents)


def from_tool_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    tool_agents = []
    for sub_agent_dict in json_dict["tool_agents"]:
        tool_agents.append(from_dict(sub_agent_dict))

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), name=json_dict["name"], tool_agents=tool_agents)


def from_dict(json_dict: dict):
    klass = globals()[json_dict["class"]]
    if "chain_agents" in json_dict:
        return from_chain_dict(json_dict)
    elif "tool_agents" in json_dict:
        return from_tool_dict(json_dict)

    config_klass = globals()[json_dict["config"]["class"]]
    return klass(config=config_klass(**json_dict["config"]), name=json_dict["name"])
