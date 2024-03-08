import json

from fastapi import FastAPI, HTTPException

from executor.core.chain_agent import *
from executor.custom.animal_info_router_agent import *
from executor.custom.asr_to_prompt_agent import *
from executor.custom.cat_facts_api_agent import *
from executor.custom.current_location_agent import *
from executor.custom.current_time_agent import *
from executor.custom.dog_facts_api_agent import *
from executor.custom.get_user_profile_agent import *
from executor.custom.response_to_asr_agent import *
from executor.custom.router_agent import *
from loader.manager import AgentManager
from loader.serialize import to_dict

app = FastAPI()

agent_manager = AgentManager()


class AsrInput(BaseModel):
    text: str = Field(description="textual input value")


class AiEgnInput(BaseModel):
    user: str = Field(description="user id")
    asr_input: AsrInput = Field(description="asr input value")


class AsrOutput(BaseModel):
    response: str = Field(description="textual output value")
    company: str = Field(description="user company")


class AiEgnOutput(BaseModel):
    asr_output: AsrOutput = Field(description="asr output value")


# Agent Executor
@app.post("/test/execute_agent/{uuid}")
def test_execute_agent(uuid: str, ai_egn_input: AiEgnInput):
    loaded_agent = agent_manager.get(uuid)
    if not loaded_agent:
        raise HTTPException(status_code=404, detail="Agent not found by uuid")

    print(f"=== Agent({uuid}) executing ===")
    start = datetime.now()
    ai_egn_output = AiEgnOutput.model_validate(loaded_agent.execute(
        user=ai_egn_input.user, agent_input={
            "asr_input": ai_egn_input.asr_input.model_dump()
        })
    )

    end = datetime.now()
    print(f"=== Agent({uuid}) executed (elapsed: {format((end - start).total_seconds(), '.3f')} s) ===")
    return ai_egn_output


# Agent Builder Mock
@app.post("/test/create_agent")
def test_create_agent(agent: dict):
    return {"uuid": agent_manager.set(agent)}


# for test debug with printing (agent serialize)
@app.get("/test/debug_agent")
def test_debug_agent():
    agent = ChainAgent(chain_agents=[               # 아래의 Agent 를 순서대로 수행하는 Agent
        AsrToPromptAgent(),                             # 1) ASR 입력을 대화 프롬프트로 변환하는 Agent
        UserProfileAgent(),                             # 2) 사용자의 이름, ip, 회사 정보를 가져오는 Agent
        RouterAgent(tool_agents=[                       # 3) 동물 정보, 사용자 위치, 현재 시각 정보를 알려줄 수 있는 추론 Agent
            AnimalInfoRouterAgent(tool_agents=[             # 동물에 대한 사실을 알려줄 수 있는 추론 Agent
                CatFactsApiAgent(),                             # 고양이에 대한 사실을 알려주는 Agent (via https://catfact.ninja)
                DogFactsApiAgent(),                             # 개에 대한 사실을 알려주는 Agent (https://dog-api.kinduff.com)
            ], config=FunctionCallConfig(system_message="You are an helpful assistant on animals.", history_limit=100)),
            CurrentLocationAgent(),                         # 사용자의 위치를 가져오는 Agent
            CurrentTimeAgent(),                             # 현재 시각을 가져오는 Agent
        ], config=FunctionCallConfig(history_limit=200)),
        ResponseToAsrAgent(),                           # 4) 대화 응답을 ASR 출력으로 변환하는 Agent
    ], config=ChainConfig(chain_map=[
        [
            AgentChain(import_key="asr_input"),
        ],
        [],
        [
            AgentChain(import_key="ip", to_key="local_ip"),
            AgentChain(import_key="prompt")
        ],
        [
            AgentChain(import_key="company"),
            AgentChain(import_key="response")
        ],
    ]))

    agent_dict = to_dict(agent)

    serialized = json.dumps(agent_dict)
    print(serialized)

    uuid = test_create_agent(agent_dict)["uuid"]
    ai_egn_output = test_execute_agent(
        uuid=uuid,
        ai_egn_input=AiEgnInput(
            user="isaac",
            asr_input=AsrInput(text="고양이에 대한 사실 3가지랑 현재 시각, 내가 있는 위치 알려줄래")
        )
    )

    return ai_egn_output
