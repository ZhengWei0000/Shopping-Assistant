from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from agent import ShoppingAssistant
from helper import create_tool_node_with_fallback
from typing import Annotated,List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    """
    Represents the state of the conversation. Holds a list of messages exchanged between the user and assistant.
    """
    messages: Annotated[list[AnyMessage], add_messages]


class ShoppingGraph:
    def __init__(self, assistant_runnable, tools_no_confirmation, tools_need_confirmation):
        """
        Initializes the ShoppingGraph, setting up the assistant, tools, and memory.
        
        :param assistant_runnable: A callable ShoppingAssistant instance for interacting with the assistant.
        :param tools_no_confirmation: List of tools that do not require user confirmation.
        :param tools_need_confirmation: List of tools that require user confirmation before execution.
        """
        self.assistant_runnable = assistant_runnable
        self.tools_no_confirmation = tools_no_confirmation
        self.tools_need_confirmation = tools_need_confirmation
        self.confirmation_tool_names = {t.name for t in tools_need_confirmation}
        self.memory = MemorySaver()  # Initialize memory for state persistence
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Builds and returns the state graph with nodes for the assistant, tools, and the routing logic.
        
        :return: A compiled state graph.
        """
        builder = StateGraph(State)

        # Add nodes to the graph
        self._add_graph_nodes(builder)

        # Add conditional edges to route based on the tool invoked
        builder.add_conditional_edges(
            "assistant", self._route_tools, ["tools_no_confirmation", "tools_need_confirmation", END]
        )

        # Set up the basic graph edges
        self._add_graph_edges(builder)

        # Compile the graph with memory and interruption logic
        return builder.compile(
            checkpointer=self.memory,
            interrupt_before=["tools_need_confirmation"]
        )

    def _add_graph_nodes(self, builder):
        """
        Add the necessary nodes (assistant and tools) to the state graph.
        
        :param builder: The StateGraph builder to which the nodes will be added.
        """
        builder.add_node("assistant", ShoppingAssistant(self.assistant_runnable))
        builder.add_node("tools_no_confirmation", create_tool_node_with_fallback(self.tools_no_confirmation))
        builder.add_node("tools_need_confirmation", create_tool_node_with_fallback(self.tools_need_confirmation))

    def _add_graph_edges(self, builder):
        """
        Add the required edges between nodes in the state graph.
        
        :param builder: The StateGraph builder to which the edges will be added.
        """
        builder.add_edge(START, "assistant")
        builder.add_edge("tools_no_confirmation", "assistant")
        builder.add_edge("tools_need_confirmation", "assistant")

    def _route_tools(self, state):
        """
        Routes the tool invocations to the appropriate node based on the tool call name.

        :param state: The current state of the conversation, containing the last message.
        :return: The next node to transition to (either tools_no_confirmation or tools_need_confirmation).
        """
        next_node = tools_condition(state)
        if next_node == END:
            return END

        ai_message = state["messages"][-1]
        first_tool_call = ai_message.tool_calls[0]
        
        if first_tool_call["name"] in self.confirmation_tool_names:
            return "tools_need_confirmation"

        return "tools_no_confirmation"

    def stream_responses(self, input_data, config):
        """
        Streams the responses from the assistant graph based on the provided input data and config.
        
        :param input_data: The input data (user's message or query).
        :param config: The configuration for the graph (such as user info, etc).
        :return: A stream of responses from the assistant.
        """
        return self.graph.stream(input_data, config, stream_mode="values")

    def get_state(self, config):
        """
        Retrieves the current state of the graph, which includes the assistant's context and conversation history.
        
        :param config: The configuration containing state identifiers like user_id.
        :return: The current state of the graph.
        """
        return self.graph.get_state(config)

    def invoke(self, input_data, config):
        """
        Directly invokes the graph with the given input data and configuration.
        
        :param input_data: The input data for the graph.
        :param config: The configuration to guide the invocation (e.g., user context).
        :return: The result of the graph invocation.
        """
        return self.graph.invoke(input_data, config)
