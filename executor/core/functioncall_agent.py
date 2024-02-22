import json
import os

from openai import OpenAI
from pydantic import Field

from executor.core.agent import Agent
from executor.core.orchestration_agent import OrchestrationAgent, OrchestrationConfig, OrchestrationAgentInput


class FunctionCallConfig(OrchestrationConfig):
    history_limit: int = Field(description="history limit", default=50)


class FunctionCallAgent(OrchestrationAgent):
    def __init__(self, config: FunctionCallConfig, tool_agents: list[Agent]=[], name: str = None):
        super().__init__(tool_agents=tool_agents, config=config, name=name)
        self.__client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.__limit = config.history_limit

    def execute(self, agent_input: dict) -> dict:
        orchestration_input = OrchestrationAgentInput.parse_obj(agent_input)
        messages = []

        memory = OrchestrationAgent.memory_manager.get_memory(self.name, orchestration_input.user)
        history = memory.get(self.__limit)
        if not history:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": orchestration_input.prompt})

        completion = self._chat_completion(messages=history + messages)

        while True:
            if completion.finish_reason == "stop":
                messages.append(completion.message)
                break
            elif completion.finish_reason == "tool_calls":
                messages.append(completion.message)
                for tool_call in completion.message.tool_calls:
                    tool = list(filter(lambda x: x.name == tool_call.function.name, self.tool_agents))[0]
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": json.dumps(tool.execute(json.loads(tool_call.function.arguments)))
                        }
                    )

            completion = self._chat_completion(messages=history + messages)

        memory.add(messages)
        return {"response": completion.message.content}

    def _chat_completion(self, messages):
        completion = self.__client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=self.tool_schemas if self.tool_schemas else None
        )

        return completion.choices[0]
