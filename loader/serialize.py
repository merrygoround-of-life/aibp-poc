from executor.core.agent import Agent, AgentType
from executor.core.chain_agent import ChainAgent
from executor.core.orchestration_agent import OrchestrationAgent


def to_chain_dict(agent: ChainAgent):
    chain_agents = []
    for sub_agent in agent.chain_agents:
        chain_agents.append(to_dict(sub_agent))
    return {"chain_agents": chain_agents}


def to_tool_dict(agent: OrchestrationAgent):
    tool_agents = []
    for sub_agent in agent.tool_agents:
        tool_agents.append(to_dict(sub_agent))
    return {"tool_agents": tool_agents}


def to_dict(agent: Agent):
    agent_dict = {
        "name": agent.name,
        "class": agent.__class__.__name__,
        "config": {
            **{"class": agent.config.__class__.__name__},
            **agent.config.dict()
        }
    }

    if agent.type == AgentType.CHAIN:
        return {**agent_dict, **to_chain_dict(agent)}
    elif agent.type == AgentType.ORCHESTRATION:
        return {**agent_dict, **to_tool_dict(agent)}

    return agent_dict
