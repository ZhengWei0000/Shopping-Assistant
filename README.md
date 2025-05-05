# Shopping-Assistant

## Prerequisites

OPENAI_API_KEY in .env
Python 3.11 or higher
SQLite
LangGraph

#### Set up a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

#### Install dependencies:

```
pip install -r requirements.txt
```

#### Run the Shopping Assistant for gradio:

```
python gradio_interface.py
```

#### Run the Shopping Assistant for Terminal:

```
python main.py
```

#### Interactive Mode:

The assistant will prompt you to type your shopping-related queries. Common operations include:

1. Searching for products by title, category, brand.
2. Adding/removing items from the cart with confirmation.
3. Viewing cart details, checking out, and retrieving payment options.

Here is some example：

User: "Please show me some available products in the beauty category."

Assistant: "Here are some available beauty products:

1. Essence Mascara Lash Princess - $9.99
2. Red Lipstick - $12.99
   ..."

User: "Add Essence Mascara Lash Princess to my cart."

Assistant: "Are you sure you want to add this product to your cart? Type 'y' to confirm or provide additional instructions."

User: "y"

Assistant: "The item has been added to your cart."
