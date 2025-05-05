from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import List, Dict, Optional


def handle_tool_error(state: Dict) -> Dict:
    """
    Handles errors encountered during tool execution. It will extract the error message from the state 
    and format it into a response message for the tool calls.

    :param state: The current state of the graph, which includes the error and tool calls.
    :return: A dictionary with error messages for each tool call.
    """
    error = state.get("error")
    tool_calls = state.get("messages", [])[-1].tool_calls  # Ensure messages exist
    error_messages = [
        ToolMessage(
            content=f"Error: {repr(error)}\nPlease fix your mistakes.",
            tool_call_id=tc["id"]
        )
        for tc in tool_calls
    ]
    return {"messages": error_messages}


def create_tool_node_with_fallback(tools: List) -> ToolNode:
    """
    Creates a tool node with fallbacks in case of errors. It will handle errors by invoking 
    `handle_tool_error` when an error occurs.

    :param tools: A list of tools to be added to the node.
    :return: A ToolNode configured with the fallback error handling.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: Dict, printed_ids: set, max_length: int = 1500) -> None:
    """
    Prints the event message if it hasn't been printed before. It also ensures the message is not too long 
    by truncating it if necessary.

    :param event: The event data containing the dialog state and messages.
    :param printed_ids: A set of already printed message IDs to avoid printing duplicates.
    :param max_length: The maximum length for the message representation before truncating.
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])

    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]  # Get the last message in the list

        if message.id not in printed_ids:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            printed_ids.add(message.id)
