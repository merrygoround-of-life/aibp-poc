class ChatHistory:
    def __init__(self):
        self.history = []

    def add(self, messages):
        self.history.extend(messages)

    def get(self, num: int):
        return self.history[-min(len(self.history), num):]


class MemoryManager:
    def __init__(self):
        self.memory: dict[str, ChatHistory] = {}

    def get_memory(self, agent: str, user: str):
        key = f"{agent}_{user}"
        if key not in self.memory:
            self.memory[key] = ChatHistory()

        return self.memory[key]
