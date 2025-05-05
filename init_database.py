import os
import shutil
import sqlite3
import pandas as pd
import requests

# Define the URL for a product API (using DummyJSON as a mock API)
api_url = "https://dummyjson.com/products"
local_file = "shop_db_data.sqlite"
backup_file = "shop_db_data.backup.sqlite"
overwrite = True

# Check if we need to create or overwrite the local SQLite database
if overwrite or not os.path.exists(local_file):
    response = requests.get(api_url)
    response.raise_for_status()  # Ensure the request was successful
    products_data = response.json().get("products", [])  # Extract the products list

    # Connect to SQLite and create tables
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    # Create a table schema for products
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        title TEXT,
        description TEXT,
        price REAL,
        discountPercentage REAL,
        rating REAL,
        stock INTEGER,
        brand TEXT,
        category TEXT,
        thumbnail TEXT
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cart (
        user_id TEXT NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id)
    )
    """)

    # Insert product data into the products table with error handling for missing fields
    for product in products_data:
        cursor.execute("""
        INSERT INTO products (id, title, description, price, discountPercentage, rating, stock, brand, category, thumbnail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product.get("id", None),
            product.get("title", "Unknown Title"),
            product.get("description", "No description available"),
            product.get("price", 0.0),
            product.get("discountPercentage", 0.0),
            product.get("rating", 0.0),
            product.get("stock", 0),
            product.get("brand", "Unknown Brand"),
            product.get("category", "Uncategorized"),
            product.get("thumbnail", "")
        ))

    # Commit the transaction and create a backup copy
    conn.commit()
    shutil.copy(local_file, backup_file)
    conn.close()

# Helper function to view product data
def view_products(file):
    conn = sqlite3.connect(file)
    df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    return df


