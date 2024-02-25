from executor.core.functioncall_agent import FunctionCallAgent, FunctionCallConfig


class AnimalInfoRouterAgent(FunctionCallAgent):
    description = "Use when you want to know the facts about an animal."

    def __init__(self, config: FunctionCallConfig, tool_agents=None, uuid=None):
        super().__init__(tool_agents=tool_agents, config=config, uuid=uuid)
