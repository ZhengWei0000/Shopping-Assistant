# coding:utf-8
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
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
    db_query
)
from graph import ShoppingGraph
import gradio as gr

load_dotenv()
thread_id = str(uuid.uuid4())
thread_id = '8b909675-76b1-4913-9f1d-e77897e016ec'

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
                "When handling product queries, ensure all parameters that are not explicitly provided by the user are set to `None` instead of an empty string. "
                "Use the appropriate tools to retrieve delivery times and payment methods. Avoid making guesses or assumptions if required database information is unavailable. "
                "If a tool returns an empty response, kindly ask the user to rephrase their question or provide additional details. "
                "Ensure that you only communicate capabilities you possess, and if any tool function returns an error, relay the error message to the user in a helpful manner."
                "\n\nWhenever you need to use a tool, respond in the following format:\nTool: tool_name(arg1=value1,arg2=value2,...)\n\nFor example:\nTool: add_to_cart(product_id=123)\n\nCurrent user:\n<User>\n{user_info}\n</User>"
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

def initialize_shopping_graph():
    llm = initialize_llm()
    assistant_prompt = create_assistant_prompt()
    tools_no_confirmation, tools_need_confirmation = configure_tools()

    assistant_runnable = assistant_prompt | llm.bind_tools(tools_no_confirmation + tools_need_confirmation)
    shopping_graph = ShoppingGraph(assistant_runnable, tools_no_confirmation, tools_need_confirmation)
    return shopping_graph

def generate_session_config():
    return {
        "configurable": {
            "user_id": thread_id,
            "thread_id": thread_id,
        }
    }

def process_user_message(shopping_graph, user_message, config, chat_history):
    chat_history.append({'role': 'user', 'content': user_message})
    continue_processing = True

    while continue_processing:
        input_data = {"messages": chat_history}
        events = shopping_graph.stream_responses(input_data, config)
        response = ""
        for event in events:
            last_message = event["messages"][-1]
            response = last_message.content
        
        # print(f"Assistant response: {response}")  # Debug
        chat_history.append({'role': 'assistant', 'content': response})

        if response.startswith("Tool: "):
            try:
                tool_call_str = response[len("Tool: "):]
                tool_name, args_str = tool_call_str.split("(", 1)
                args_str = args_str.rstrip(")")
                args = {}
                for arg in args_str.split(","):
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        args[key.strip()] = value.strip()
                
                if tool_name == "add_to_cart":
                    product_id = args.get("product_id")
                    add_to_cart.invoke({'input': {'config': config, 'product_id': product_id, 'quantity': 1}})
                    tool_response = f"Item {product_id} has been added to the cart."
                elif tool_name == "remove_from_cart":
                    product_id = args.get("product_id")
                    remove_from_cart.invoke({'input': {'config': config, 'product_id': product_id}})
                    tool_response = f"Item {product_id} has been removed from the cart."
                else:
                    tool_response = f"Unknown tool call: {tool_name}"
                
                chat_history.append({'role': 'tool', 'content': tool_response})
            except Exception as e:
                tool_response = f"Error while processing tool call: {str(e)}"
                chat_history.append({'role': 'assistant', 'content': tool_response})
                continue_processing = False
        else:
            snapshot = shopping_graph.get_state(config)
            while snapshot.next:
                confirmation_input = "y"
                if confirmation_input.strip().lower() == "y":
                    result = shopping_graph.invoke(None, config)
                    if result and 'messages' in result and len(result['messages']) > 0:
                        confirmation_response = result['messages'][-1].content
                        chat_history.append({'role': 'assistant', 'content': confirmation_response})
                snapshot = shopping_graph.get_state(config)
            continue_processing = False
    return chat_history


def gradio_chat(user_message, chat_history, session_state):
    if session_state.get("user_id") is None:
        config = generate_session_config()
        session_state["user_id"] = config["configurable"]["user_id"]
        shopping_graph = initialize_shopping_graph()
        session_state["shopping_graph"] = shopping_graph
        session_state["config"] = config
    else:
        shopping_graph = session_state["shopping_graph"]
        config = session_state["config"]


    updated_chat_history = process_user_message(shopping_graph, user_message, config, chat_history)
    cart_html = generate_cart_html(config)
    return updated_chat_history, session_state, cart_html

def clear_chat():
    cart_html = ''
    return [], {"user_id": None, "shopping_graph": None, "config": None}, cart_html

def initial_load(session_state):
    if session_state.get("user_id") is None:
        config = generate_session_config()
        session_state["user_id"] = config["configurable"]["user_id"]
        shopping_graph = initialize_shopping_graph()
        session_state["shopping_graph"] = shopping_graph
        session_state["config"] = config
    else:
        shopping_graph = session_state["shopping_graph"]
        config = session_state["config"]

    cart_html = generate_cart_html(config)
    chat_history = [
        {'role': 'assistant', 'content': 'Welcome to Shopping Assistant, Please show me some available products and category.'}
    ]

    return chat_history, session_state,cart_html

def generate_cart_html(config):
    user_id = config.get("configurable", {}).get("thread_id", None)
    if not user_id:
        raise ValueError("No user_id configured.")

    query = """
        SELECT p.id as product_id, p.title, p.price, c.quantity,p.thumbnail as image
        FROM cart c 
        JOIN products p ON c.product_id = p.id
        WHERE c.user_id = ?
    """
    cart_items = db_query(query, (user_id,))

    total_price = sum(item["price"] * item["quantity"] for item in cart_items)
    items = [{"product_id": item["product_id"], "title": item["title"], "price": item["price"], "quantity": item["quantity"],"image": item["image"]} for item in cart_items]

    if not items:
        return "<h3>Shopping Cart</h3><p>The shopping cart is empty.</p>"
    html = "<h3>My Shopping Cart</h3><ul>"
    for item in items:
        html += f"""
            <li style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="{item['image']}" width="80" height="80" style="margin-right: 10px;">
                <div>
                    <p style="margin: 0;"><strong>{item['title']}</strong></p>
                    <p style="margin: 0;">{item['price']} x {item['quantity']}</p>
                </div>
            </li>
        """
    html += "</ul>"
    html +=f"<h3>Total Price: {round(total_price,4)}$</h3>"
    return html

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("<h1>Shopping Assistant</h1>")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="chat", type="messages")
                with gr.Row():
                    user_input = gr.Textbox(show_label=False, placeholder="Enter the message and press Enter")
            
            with gr.Column(scale=1):
                cart_display = gr.HTML("<h3>Shopping cart</h3><p>Shopping cart is empty.</p>")

        session = gr.State({"user_id": None, "shopping_graph": None, "config": None, "cart": [], "recommendations": []})

        user_input.submit(
            fn=gradio_chat,
            inputs=[user_input, chatbot, session],
            outputs=[chatbot, session, cart_display]
        )

        clear_btn = gr.Button("Clear chat history")
        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, session, cart_display]
        )

        demo.load(
            fn=initial_load,
            inputs=[session],
            outputs=[chatbot, session, cart_display]
        )

    demo.launch(share=True)


if __name__ == "__main__":
    create_gradio_interface()
