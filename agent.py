from langchain_core.runnables import RunnableConfig
from typing import Dict


class ShoppingAssistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: Dict, config: RunnableConfig):
        """
        Execute the main logic of the assistant until a valid response is obtained
        """
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)

            state = {**state, "user_info": user_id}

            result = self.runnable.invoke(state)

            if self._is_invalid_result(result):
                state = self._re_prompt(state)
            else:
                break

        return {"messages": result}

    def _is_invalid_result(self, result):
        return not result.tool_calls and (
            not result.content or (isinstance(result.content, list) and not result.content[0].get("text"))
        )

    def _re_prompt(self, state: Dict):
        messages = state.get("messages", []) + [("user", "Please provide a detailed response.")]
        return {**state, "messages": messages}
