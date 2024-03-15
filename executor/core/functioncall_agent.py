import json
import os

from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from pydantic import Field

from executor.core.agent import AgentConfig
from executor.core.orchestration_agent import OrchestrationAgent, OrchestrationAgentInput


class FunctionCallConfig(AgentConfig):
    model: str = Field(description="model to use", default="gpt-3.5-turbo")
    system_message: str = Field(description="system message", default="You are a helpful assistant.")
    history_limit: int = Field(description="history limit", default=50)


class FunctionCallAgent(OrchestrationAgent):
    def __init__(self, config: FunctionCallConfig, tool_agents=None, uuid=None):
        super().__init__(tool_agents=tool_agents, config=config, uuid=uuid)
        self.__client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.__model = config.model
        self.__system_message = FunctionCallConfig.model_validate(config).system_message
        self.__limit = config.history_limit

    def invoke(self, user: str, agent_input: dict) -> dict:
        orchestration_input = OrchestrationAgentInput.model_validate(agent_input)

        messages = [ChatCompletionSystemMessageParam(
            role="system",
            content=self.__system_message
        )]

        memory = OrchestrationAgent.memory_manager.get_memory(self.uuid, user)
        history = memory.get(self.__limit)
        messages.extend(history)

        for message in self.get_additional_messages(agent_input):
            messages.append(message)

        messages.append(ChatCompletionUserMessageParam(
            role="user",
            content=orchestration_input.prompt
        ))

        reason_stop = False
        while not reason_stop:
            stream_completion = self.reason(messages=messages)

            choice_full = None
            for chunk_completion in stream_completion:
                completion = chunk_completion.choices[0]
                if not choice_full:
                    choice_full = completion.delta
                else:
                    _merge_choice_delta(choice_full, completion.delta)

                if completion.finish_reason == "stop":
                    messages.append(choice_full)
                    reason_stop = True
                elif completion.finish_reason == "tool_calls":
                    messages.append(choice_full)
                    for tool_call in choice_full.tool_calls:
                        tool = list(filter(lambda x: x.uuid == tool_call.function.name, self.tool_agents))[0]
                        tool_output = tool.execute(user, json.loads(tool_call.function.arguments))
                        messages.append(ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=json.dumps(tool_output)
                        ))

        memory.add(messages)
        return {"response": choice_full.content}

    def reason(self, messages):
        print(f"[{self.__class__.__name__}] reasoning..")

        return self.__client.chat.completions.create(
            model=self.__model,
            messages=messages,
            stream=True,
            tools=self.tool_schemas if self.tool_schemas else None
        )

    def parameters_schema(self) -> dict:
        return OrchestrationAgentInput.schema()


def _merge_choice_delta(full: ChoiceDelta, delta: ChoiceDelta):
    def merge_tool_calls(tc_fulls: list[ChoiceDeltaToolCall], tc_deltas: list[ChoiceDeltaToolCall]):
        def merge_tool_call_function(tcf_full: ChoiceDeltaToolCallFunction, tcf_delta: ChoiceDeltaToolCallFunction):
            if tcf_delta.name:
                tcf_full.name = tcf_delta.name
            if tcf_delta.arguments:
                tcf_full.arguments = tcf_full.arguments + tcf_delta.arguments if tcf_full.arguments else tcf_delta.arguments

        for tc_delta in tc_deltas:
            index = tc_delta.index
            if index >= len(tc_fulls):
                tc_fulls.append(tc_delta)   # assume that the indexes appear in order.
            if tc_delta.id:
                tc_fulls[index].id = tc_fulls[index].id + tc_delta.id if tc_fulls[index].id else tc_delta.id
            if tc_delta.type:
                tc_fulls[index].type = tc_delta.type
            if tc_delta.function:
                if not tc_fulls[index].function:
                    tc_fulls[index].function = tc_delta.function
                else:
                    merge_tool_call_function(tc_fulls[index].function, tc_delta.function)

    if delta.role:
        full.role = delta.role
    if delta.content:
        full.content = full.content + delta.content if full.content else delta.content
    if delta.tool_calls:
        if not full.tool_calls:
            full.tool_calls = delta.tool_calls
        else:
            merge_tool_calls(full.tool_calls, delta.tool_calls)
