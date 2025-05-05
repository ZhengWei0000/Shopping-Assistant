import sqlite3
from typing import List, Dict, Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from datetime import datetime, timedelta

db = "shop_db_data.sqlite"


def db_query(query: str, params: tuple = ()) -> List[Dict]:
    """Helper function to execute queries and return results."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]
    except Exception as e:
        return [{"message": f"An error occurred: {str(e)}"}]
    finally:
        cursor.close()
        conn.close()
    return results


@tool
def fetch_product_by_title(title: str) -> List[Dict]:
    """Fetches up to 10 products by title."""
    query = """
    SELECT id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail 
    FROM products
    WHERE title LIKE ? LIMIT 10
    """
    results = db_query(query, (f"%{title}%",))
    if not results:
        return [{"message": "No products found with the specified title."}]
    return results


@tool
def fetch_product_by_category(category: str) -> List[Dict]:
    """Fetches up to 10 products by category."""
    query = """
    SELECT id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail 
    FROM products
    WHERE category = ? LIMIT 10
    """
    results = db_query(query, (category,))
    if not results:
        return [{"message": "No products found in the specified category."}]
    return results


@tool
def fetch_product_by_brand(brand: str) -> List[Dict]:
    """Fetches up to 10 products by brand."""
    query = """
    SELECT id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail 
    FROM products
    WHERE brand = ? LIMIT 10
    """
    results = db_query(query, (brand,))
    if not results:
        return [{"message": "No products found for the specified brand."}]
    return results


@tool
def initialize_fetch() -> List[Dict]:
    """Fetches information on a limited number of available products."""
    query = """
    SELECT id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail 
    FROM products
    LIMIT 10
    """
    return db_query(query)


@tool
def fetch_all_categories() -> List[str]:
    """Fetches all unique product categories from the database."""
    query = "SELECT DISTINCT category FROM products ORDER BY category"
    categories = db_query(query)
    if not categories:
        return [{"message": "No categories found."}]
    return [category["category"] for category in categories]


@tool
def fetch_recommendations(input: Dict) -> List[Dict]:
    """Fetch similar products based on content-based filtering (category and brand)."""
    # Fetch category and brand for the given product_id
    product_id = input.get('product_id')
    if not product_id:
        return [{"message": "NO product_id."}]
    
    # 其余推荐逻辑
    query = """
    SELECT id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail 
    FROM products
    WHERE (category = (SELECT category FROM products WHERE id = ?) OR brand = (SELECT brand FROM products WHERE id = ?)) 
    AND id != ? 
    LIMIT 5
    """
    results = db_query(query, (product_id, product_id, product_id))
    if not results:
        return [{"message": "Not found similar products."}]
    return results

@tool
def add_to_cart(config: RunnableConfig, product_id: int, quantity: int = 1) -> Dict:
    """Adds an item to the user's cart, checks if it's out of stock or if stock is insufficient, and provides a confirmation message."""
    try:
        user_id = config.get("configurable", {}).get("thread_id", None)
        if not user_id:
            raise ValueError("No user_id configured.")
        
        # Check stock availability for the requested product
        query = "SELECT stock FROM products WHERE id = ?"
        stock_result = db_query(query, (product_id,))
        if not stock_result:
            return {"message": "Product not found."}
        
        available_stock = stock_result[0]["stock"]
        if available_stock < quantity:
            return {"message": f"Insufficient stock. Only {available_stock} items are available."}

        # Check if the item is already in the cart
        query = "SELECT quantity FROM cart WHERE user_id = ? AND product_id = ?"
        cart_item = db_query(query, (user_id, product_id))

        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        if cart_item:
            # Update quantity if the item is already in the cart
            new_quantity = cart_item[0]["quantity"] + quantity
            cursor.execute("UPDATE cart SET quantity = ? WHERE user_id = ? AND product_id = ?", (new_quantity, user_id, product_id))
            action = "updated"
        else:
            # Add new item to the cart
            cursor.execute("INSERT INTO cart (user_id, product_id, quantity) VALUES (?, ?, ?)", (user_id, product_id, quantity))
            action = "added"
        
        conn.commit()

        # Fetch the updated cart for confirmation
        cursor.execute("SELECT product_id, quantity FROM cart WHERE user_id = ?", (user_id,))
        cart_items = cursor.fetchall()

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

    return {
        "message": f"Item has been {action} in your cart.",
        "cart": [{"product_id": item[0], "quantity": item[1]} for item in cart_items]
    }


@tool
def remove_from_cart(config: RunnableConfig, product_id: int) -> Dict:
    """Removes an item from the user's cart, asks for confirmation, and provides a final message."""
    try:
        user_id = config.get("configurable", {}).get("thread_id", None)
        if not user_id:
            raise ValueError("No user_id configured.")
        
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Check if the item exists in the cart
        query = "SELECT quantity FROM cart WHERE user_id = ? AND product_id = ?"
        result = db_query(query, (user_id, product_id))

        if not result:
            return {"message": "Item not found in your cart."}

        # Remove the item from the cart
        cursor.execute("DELETE FROM cart WHERE user_id = ? AND product_id = ?", (user_id, product_id))
        conn.commit()

        # Fetch the updated cart for confirmation
        cursor.execute("SELECT product_id, quantity FROM cart WHERE user_id = ?", (user_id,))
        cart_items = cursor.fetchall()

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

    return {
        "message": "Item has been removed from your cart."
    }


@tool
def view_checkout_info(config: RunnableConfig) -> Dict:
    """Provides a summary of items in the cart for the given user, including total price for checkout."""
    try:
        user_id = config.get("configurable", {}).get("thread_id", None)
        if not user_id:
            raise ValueError("No user_id configured.")

        query = """
            SELECT p.id as product_id, p.title, p.price, c.quantity 
            FROM cart c 
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = ?
        """
        cart_items = db_query(query, (user_id,))

        total_price = sum(item["price"] * item["quantity"] for item in cart_items)
        items = [{"product_id": item["product_id"], "title": item["title"], "price": item["price"], "quantity": item["quantity"]} for item in cart_items]

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
    return {
        "message": "Checkout summary:",
        "total_price": total_price,
        "items": items
    }


@tool
def get_delivery_estimate() -> Dict:
    """Provides a generic estimated delivery time for an order."""
    estimated_delivery = datetime.now() + timedelta(days=5)
    return {
        "message": "Estimated delivery time:",
        "delivery_estimate": estimated_delivery.strftime('%Y-%m-%d')
    }


@tool
def get_payment_options() -> Dict:
    """Provides available payment options for the user."""
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Gift Card"]
    return {
        "message": "Available payment options:",
        "payment_options": payment_methods
    }
