# coding:utf-8
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools import (
    fetch_product_by_title,
    fetch_product_by_category,
    fetch_product_by_brand,
    initialize_fetch,
    fetch_all_categories,
    fetch_recommendations,
    add_to_cart,
    remove_from_cart,
    view_checkout_info,
    get_delivery_estimate,
    get_payment_options,
)
from graph import ShoppingGraph
from helper import _print_event

load_dotenv()

def initialize_llm():
    return ChatOpenAI(
        temperature=0.95,
        model="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def create_assistant_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful shopping assistant dedicated to providing accurate and friendly responses. "
                "Use the available tools to answer product queries, recommend items, manage the shopping cart, and provide checkout information, delivery times, and payment options. "
                "Always ensure that all product, availability, and price information is sourced from the database, "
                "When handling product queries, ensure all parameters that are not explicitly provided by the user are set to `None` instead of an empty string."
                "and use the appropriate tools to retrieve delivery times and payment methods. Avoid making guesses or assumptions if required database information is unavailable. "
                "If a tool returns an empty response, kindly ask the user to rephrase their question or provide additional details. "
                "Ensure that you only communicate capabilities you possess, and if any tool function returns an error, relay the error message to the user in a helpful manner."
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

def configure_tools():
    tools_no_confirmation = [
        fetch_product_by_title,
        fetch_product_by_category,
        fetch_product_by_brand,
        initialize_fetch,
        fetch_all_categories,
        fetch_recommendations,
        view_checkout_info,
        get_delivery_estimate,
        get_payment_options,
    ]

    tools_need_confirmation = [add_to_cart, remove_from_cart]

    return tools_no_confirmation, tools_need_confirmation

def initialize_shopping_graph(assistant_runnable, tools_no_confirmation, tools_need_confirmation):
    return ShoppingGraph(assistant_runnable, tools_no_confirmation, tools_need_confirmation)

def generate_session_config():
    thread_id = str(uuid.uuid4())
    return {
        "configurable": {
            "user_id": thread_id,
            "thread_id": thread_id,
        }
    }

def start_interactive_session():
    print("Please wait for initialization...")

    llm = initialize_llm()
    assistant_prompt = create_assistant_prompt()
    tools_no_confirmation, tools_need_confirmation = configure_tools()
    
    assistant_runnable = assistant_prompt | llm.bind_tools(tools_no_confirmation + tools_need_confirmation)
    shopping_graph = initialize_shopping_graph(assistant_runnable, tools_no_confirmation, tools_need_confirmation)
    config = generate_session_config()

    initial_query = "Please welcome me, and show me some available products and category."
    initial_events = shopping_graph.stream_responses({"messages": ("user", initial_query)}, config)

    for event in initial_events:
        final_result = event

    final_result["messages"][-1].pretty_print()

    print("\nType your question below (or type 'exit' to end):\n")

    while True:
        question = input("\nYou: ")

        if question.lower() == 'exit':
            print("Ending session. Thank you for using the shopping assistant!")
            break

        events = shopping_graph.stream_responses({"messages": ("user", question)}, config)
        for event in events:
            _print_event(event, set())

        snapshot = shopping_graph.get_state(config)
        while snapshot.next:
            try:
                user_input = input("\nAre you sure about that? Type 'y' to continue; otherwise, explain your requested changes.\n")
            except:
                user_input = "y"
            if user_input.strip().lower() == "y":
                result = shopping_graph.invoke(None, config)
                print(result['messages'][-1].content)
            else:
                result = shopping_graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
            snapshot = shopping_graph.get_state(config)

if __name__ == "__main__":
    start_interactive_session()
