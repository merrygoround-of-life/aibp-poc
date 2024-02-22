from fastapi import FastAPI

from executor.core.chain_agent import *
from executor.core.functioncall_agent import *
from executor.custom.cat_facts_agent import *
from executor.custom.get_current_time_agent import *
from executor.custom.get_user_profile_agent import *
from executor.custom.get_location_agent import *
from loader.manager import AgentManager
from loader.serialize import to_dict

app = FastAPI()

agent_manager = AgentManager()


# Agent Executor
@app.post("/test/execute_agent/{name}")
def test_execute_agent(name: str, data: dict):
    execute_output = {}

    loaded_agent = agent_manager.get(name)
    if loaded_agent:
        execute_output = loaded_agent.execute(data)

    return execute_output


# Agent Builder Mock
@app.post("/test/create_agent")
def test_create_agent(agent: dict):
    agent_manager.set(agent)


# for test debug with printing (agent serialize)
@app.get("/test/debug_agent")
def test_debug_agent():
    agent = ChainAgent(chain_agents=[   # 아래의 Agent 를 순서대로 수행하는 Agent
        UserProfileAgent(),                 # 사용자의 이름을 가져오는 Agent
        FunctionCallAgent(tool_agents=[     # 사용자의 이름과 입력 프롬프트를 사용하여 아래의 Agent를 툴로 사용하여 추론을 실행하는 Agent
            LocationAgent(),                    # 사용자의 위치를 가져오는 Agent
            CurrentTimeAgent(),                 # 현재 시각을 가져오는 Agent
            CatFactsAgent()                     # 고양이에 대한 사실을 알려주는 Agent (via https://catfact.ninja)
        ], config=FunctionCallConfig(history_limit=100))
    ], config=ChainConfig(chain_map=[
        [],
        [
            AgentChain(from_key="user", to_key="user"),
            AgentChain(from_key="prompt", to_key="prompt")
        ],
    ]))

    serialized = json.dumps(to_dict(agent))
    print(serialized)

    execute_output = agent.execute({"prompt": "고양이에 대한 사실 3가지랑 현재 시각, 내가 있는 위치 알려줄래"})
    return execute_output
