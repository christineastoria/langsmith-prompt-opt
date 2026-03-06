"""
Sets up the shopping assistant's data layer:
  1. SQLite database (products, inventory, price history, users, orders, reviews, cart)
  2. Chroma vector store (semantic product search)

Run once before starting the assistant:
    uv run python src/setup_db.py
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "shop.db"
CHROMA_PATH = ROOT / "data" / "chroma"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sku         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    brand       TEXT NOT NULL,
    category    TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    price       REAL NOT NULL,
    description TEXT NOT NULL,
    specs       TEXT NOT NULL  -- JSON
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id    INTEGER PRIMARY KEY REFERENCES products(id),
    quantity      INTEGER NOT NULL,
    warehouse     TEXT NOT NULL,
    last_updated  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS price_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER REFERENCES products(id),
    price       REAL NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    preferences TEXT NOT NULL  -- JSON: {"sizes": {...}, "brands": [...], "categories": [...]}
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    status      TEXT NOT NULL,  -- pending, processing, shipped, delivered, returned, cancelled
    created_at  TEXT NOT NULL,
    total       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id            INTEGER REFERENCES orders(id),
    product_id          INTEGER REFERENCES products(id),
    quantity            INTEGER NOT NULL,
    price_at_purchase   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER REFERENCES products(id),
    user_id     INTEGER REFERENCES users(id),
    rating      INTEGER NOT NULL,  -- 1-5
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cart (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    product_id  INTEGER REFERENCES products(id),
    quantity    INTEGER NOT NULL DEFAULT 1,
    added_at    TEXT NOT NULL,
    UNIQUE(user_id, product_id)
);

CREATE TABLE IF NOT EXISTS wishlist (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    product_id  INTEGER REFERENCES products(id),
    added_at    TEXT NOT NULL,
    UNIQUE(user_id, product_id)
);
"""


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

PRODUCTS = [
    # --- Electronics ---
    {
        "sku": "ELEC-001",
        "name": "MacBook Pro 14-inch M4",
        "brand": "Apple",
        "category": "Electronics",
        "subcategory": "Laptops",
        "price": 1599.00,
        "description": "Apple's 14-inch MacBook Pro with M4 chip. Exceptional performance for creative professionals, with up to 24GB unified memory and a stunning Liquid Retina XDR display.",
        "specs": {"storage": "512GB SSD", "memory": "16GB", "display": "14.2-inch Liquid Retina XDR", "battery": "Up to 24 hours", "ports": ["Thunderbolt 4 x3", "HDMI", "SD card", "MagSafe 3"]},
    },
    {
        "sku": "ELEC-002",
        "name": "Dell XPS 13",
        "brand": "Dell",
        "category": "Electronics",
        "subcategory": "Laptops",
        "price": 1299.00,
        "description": "Dell's ultra-thin XPS 13 with Intel Core Ultra 7. Compact, premium build quality with an InfinityEdge display and all-day battery life. Great for travel and everyday productivity.",
        "specs": {"storage": "512GB SSD", "memory": "16GB LPDDR5", "display": "13.4-inch OLED touch", "battery": "Up to 13 hours", "ports": ["Thunderbolt 4 x2", "USB-C", "microSD"]},
    },
    {
        "sku": "ELEC-003",
        "name": "Sony WH-1000XM6",
        "brand": "Sony",
        "category": "Electronics",
        "subcategory": "Headphones",
        "price": 349.00,
        "description": "Sony's flagship noise-cancelling wireless headphones. Industry-leading ANC with Multipoint Bluetooth, 30-hour battery, and exceptional sound quality. Perfect for commuting and travel.",
        "specs": {"type": "Over-ear", "connectivity": "Bluetooth 5.3", "battery": "30 hours ANC on", "anc": "Yes - Adaptive", "weight": "250g", "foldable": True},
    },
    {
        "sku": "ELEC-004",
        "name": "Apple AirPods Pro (3rd Gen)",
        "brand": "Apple",
        "category": "Electronics",
        "subcategory": "Headphones",
        "price": 249.00,
        "description": "Apple's premium in-ear wireless earbuds with H2 chip. Best-in-class ANC, transparency mode, and Adaptive Audio. Seamless integration with Apple devices.",
        "specs": {"type": "In-ear", "connectivity": "Bluetooth 5.3", "battery": "6 hours (30 with case)", "anc": "Yes - Adaptive", "water_resistance": "IPX4", "spatial_audio": True},
    },
    {
        "sku": "ELEC-005",
        "name": "Samsung 65-inch QLED 4K TV",
        "brand": "Samsung",
        "category": "Electronics",
        "subcategory": "TVs",
        "price": 899.00,
        "description": "Samsung 65-inch QLED 4K smart TV with Quantum Dot technology. Vivid colors, 120Hz refresh rate, and Gaming Hub built in. Excellent picture quality for movies and sports.",
        "specs": {"size": "65 inches", "resolution": "4K UHD", "hdr": "Quantum HDR", "refresh_rate": "120Hz", "smart_tv": "Tizen OS", "gaming_mode": True},
    },
    {
        "sku": "ELEC-006",
        "name": "iPad Pro 13-inch M4",
        "brand": "Apple",
        "category": "Electronics",
        "subcategory": "Tablets",
        "price": 1099.00,
        "description": "The most powerful iPad ever with M4 chip and Ultra Retina XDR OLED display. Incredibly thin at 5.1mm. Supports Apple Pencil Pro for creative work.",
        "specs": {"storage": "256GB", "display": "13-inch Ultra Retina XDR OLED", "chip": "Apple M4", "connectivity": "WiFi 6E + optional 5G", "pencil_support": "Apple Pencil Pro"},
    },
    # --- Footwear ---
    {
        "sku": "FOOT-001",
        "name": "Nike Air Max 270",
        "brand": "Nike",
        "category": "Footwear",
        "subcategory": "Sneakers",
        "price": 150.00,
        "description": "Nike's lifestyle sneaker featuring the largest Air unit ever. Plush cushioning and a sleek silhouette. Great for casual wear and light activity.",
        "specs": {"sole": "Max Air unit", "upper": "Mesh and synthetic", "closure": "Lace-up", "available_sizes": "US 6-15", "colorways": ["Black/White", "White/Blue", "Triple White", "Coral Pink"]},
    },
    {
        "sku": "FOOT-002",
        "name": "Adidas Ultraboost 24",
        "brand": "Adidas",
        "category": "Footwear",
        "subcategory": "Running",
        "price": 190.00,
        "description": "Adidas's premium running shoe with Boost midsole technology. Energy return with every step, Primeknit upper for a sock-like fit. Equally at home on a run or city streets.",
        "specs": {"sole": "Continental Rubber outsole + Boost midsole", "upper": "Primeknit+", "drop": "10mm", "weight": "310g (men's US 9)", "available_sizes": "US 5-14"},
    },
    {
        "sku": "FOOT-003",
        "name": "Allbirds Tree Runner",
        "brand": "Allbirds",
        "category": "Footwear",
        "subcategory": "Casual",
        "price": 128.00,
        "description": "Allbirds' beloved everyday sneaker made from sustainable eucalyptus tree fiber. Lightweight, breathable, and machine-washable. Low carbon footprint.",
        "specs": {"material": "TENCEL Lyocell from eucalyptus", "sole": "Bio-based SweetFoam", "weight": "230g (men's US 9)", "machine_washable": True, "available_sizes": "US 5-14"},
    },
    {
        "sku": "FOOT-004",
        "name": "Timberland 6-Inch Premium Waterproof Boot",
        "brand": "Timberland",
        "category": "Footwear",
        "subcategory": "Boots",
        "price": 230.00,
        "description": "The iconic Timberland boot. Waterproof full-grain leather, lug outsole, and anti-fatigue technology. Built for rugged conditions and city streets alike.",
        "specs": {"material": "Full-grain waterproof leather", "waterproofing": "Seam-sealed", "sole": "Lug rubber outsole", "insole": "Anti-fatigue technology", "available_sizes": "US 7-15"},
    },
    {
        "sku": "FOOT-005",
        "name": "New Balance 990v6",
        "brand": "New Balance",
        "category": "Footwear",
        "subcategory": "Running",
        "price": 185.00,
        "description": "New Balance's heritage running shoe, made in the USA. ENCAP midsole technology, pig suede and mesh upper. A cult classic for runners and sneaker enthusiasts.",
        "specs": {"sole": "ENCAP midsole + blown rubber outsole", "upper": "Pig suede + mesh", "made_in": "USA", "available_sizes": "US 7-20 (including wide)"},
    },
    # --- Clothing ---
    {
        "sku": "CLTH-001",
        "name": "Patagonia Nano Puff Jacket",
        "brand": "Patagonia",
        "category": "Clothing",
        "subcategory": "Jackets",
        "price": 249.00,
        "description": "Patagonia's best-selling lightweight insulated jacket. PrimaLoft Gold Eco insulation, wind and water-resistant shell. Packable into its own pocket. Fair Trade certified.",
        "specs": {"insulation": "PrimaLoft Gold Eco 60g", "shell": "Pertex Quantum", "weight": "298g (men's M)", "packable": True, "available_sizes": "XS-3XL", "fair_trade": True},
    },
    {
        "sku": "CLTH-002",
        "name": "Levi's 501 Original Jeans",
        "brand": "Levi's",
        "category": "Clothing",
        "subcategory": "Denim",
        "price": 79.00,
        "description": "The original straight-leg jeans that started it all. 100% cotton denim, button fly, classic five-pocket styling. A timeless wardrobe essential.",
        "specs": {"material": "100% cotton denim", "fit": "Straight", "closure": "Button fly", "available_sizes": "Waist 28-44, Inseam 28-34", "wash_options": ["Medium Stonewash", "Dark Rinse", "Light Destroy", "Black"]},
    },
    {
        "sku": "CLTH-003",
        "name": "Arc'teryx Beta AR Jacket",
        "brand": "Arc'teryx",
        "category": "Clothing",
        "subcategory": "Jackets",
        "price": 799.00,
        "description": "Arc'teryx's most versatile hardshell. GORE-TEX Pro 3L construction, waterproof and breathable for all-weather alpine performance. Engineered for climbers and mountaineers.",
        "specs": {"shell": "GORE-TEX Pro 3L", "weight": "485g (men's M)", "waterproof": "Completely seam-taped", "available_sizes": "XS-3XL", "helmet_compatible": True},
    },
    {
        "sku": "CLTH-004",
        "name": "Uniqlo Ultra Light Down Jacket",
        "brand": "Uniqlo",
        "category": "Clothing",
        "subcategory": "Jackets",
        "price": 89.00,
        "description": "Uniqlo's ultra-lightweight packable down jacket. 90% down fill, packable into a compact pouch. Exceptional warmth-to-weight ratio at an accessible price.",
        "specs": {"fill": "90% down, 10% feather", "weight": "195g (men's M)", "packable": True, "available_sizes": "XS-3XL"},
    },
    # --- Beauty & Wellness ---
    {
        "sku": "BEAU-001",
        "name": "Diptyque Baies Candle",
        "brand": "Diptyque",
        "category": "Beauty & Wellness",
        "subcategory": "Candles",
        "price": 75.00,
        "description": "Diptyque's most iconic candle. A bouquet of blackcurrant leaves and Bulgarian roses — fresh, green, and unmistakably Diptyque. Burns for approximately 60 hours. A staple in every NYC apartment.",
        "specs": {"size": "190g", "burn_time": "60 hours", "fragrance_notes": {"top": "Blackcurrant leaf", "heart": "Rose", "base": "Woody"}, "wax": "Vegetable and paraffin blend"},
    },
    {
        "sku": "BEAU-002",
        "name": "Diptyque Feu de Bois Candle",
        "brand": "Diptyque",
        "category": "Beauty & Wellness",
        "subcategory": "Candles",
        "price": 75.00,
        "description": "Diptyque's cozy fireside candle. The smoky warmth of a crackling wood fire in a Parisian apartment. Perfect for winter. One of Diptyque's oldest and most beloved scents.",
        "specs": {"size": "190g", "burn_time": "60 hours", "fragrance_notes": {"top": "Smoke", "heart": "Wood", "base": "Resinous"}, "wax": "Vegetable and paraffin blend"},
    },
    {
        "sku": "BEAU-003",
        "name": "Rhode Peptide Lip Treatment",
        "brand": "Rhode",
        "category": "Beauty & Wellness",
        "subcategory": "Skincare",
        "price": 20.00,
        "description": "The lip treatment that broke the internet. Hailey Bieber's Rhode Peptide Lip Treatment with collagen-boosting peptides, shea butter, and baobab oil. Sheer tint with serious glow. Sold out constantly.",
        "specs": {"shades": ["Unscented", "Salted Caramel", "Watermelon Glaze", "Espresso", "Strawberry Glaze"], "key_ingredients": ["Collagen-boosting peptides", "Shea butter", "Baobab oil"], "size": "10g"},
    },
    {
        "sku": "BEAU-004",
        "name": "Drunk Elephant Protini Polypeptide Cream",
        "brand": "Drunk Elephant",
        "category": "Beauty & Wellness",
        "subcategory": "Skincare",
        "price": 68.00,
        "description": "Drunk Elephant's cult protein moisturizer. Signal peptides, growth factors, and pygmy waterlily work together to firm and smooth skin. Lightweight but deeply nourishing. A Sephora perennial bestseller.",
        "specs": {"size": "50ml", "skin_type": "All skin types", "key_ingredients": ["Signal peptides", "Growth factors", "Amino acids", "Pygmy waterlily"], "fragrance_free": True},
    },
    {
        "sku": "BEAU-005",
        "name": "Glossier You Perfume",
        "brand": "Glossier",
        "category": "Beauty & Wellness",
        "subcategory": "Fragrance",
        "price": 72.00,
        "description": "Glossier's skin-like, musky fragrance designed to smell different on everyone. Ambrette, orris, ambrox, and sandalwood warm to your own skin chemistry. The scent that made everyone in NYC smell the same (in the best way).",
        "specs": {"size": "50ml EDP", "fragrance_notes": {"top": "Pink pepper", "heart": "Ambrette, Orris", "base": "Ambrox, Musk, Sandalwood"}, "longevity": "6-8 hours"},
    },
    # --- Activewear ---
    {
        "sku": "ACTV-001",
        "name": "Alo Yoga Airlift High-Waist Legging",
        "brand": "Alo Yoga",
        "category": "Activewear",
        "subcategory": "Leggings",
        "price": 128.00,
        "description": "Alo's best-selling performance legging. Airlift fabric is ultra-lightweight, sweat-wicking, and squat-proof with four-way stretch. High-rise waist stays put through any workout. The legging for Pilates, yoga, or just looking put-together.",
        "specs": {"material": "Airlift (87% polyester, 13% spandex)", "rise": "High-waist", "available_sizes": "2XS-3XL", "pockets": True, "colorways": ["Black", "White", "Espresso", "Ivory", "Dark Caramel"]},
    },
    {
        "sku": "ACTV-002",
        "name": "Alo Yoga Accolade Sweatshirt",
        "brand": "Alo Yoga",
        "category": "Activewear",
        "subcategory": "Tops",
        "price": 118.00,
        "description": "Alo's ultra-soft everyday sweatshirt. Relaxed cropped fit, slightly oversized with a fleece interior. The perfect layer after a workout or for a coffee run. Goes with everything.",
        "specs": {"material": "Brushed fleece (65% cotton, 35% polyester)", "fit": "Relaxed crop", "available_sizes": "XS-XL", "colorways": ["White", "Black", "Ivory", "Alosoft Powder Blue"]},
    },
    # --- Fashion ---
    {
        "sku": "FASH-001",
        "name": "Skims Fits Everybody Bodysuit",
        "brand": "Skims",
        "category": "Fashion",
        "subcategory": "Bodysuits",
        "price": 62.00,
        "description": "Skims' most popular bodysuit. The Fits Everybody fabric stretches to 3x its size to fit every body. Barely-there feel with a smooth, seamless look under clothes. A wardrobe essential.",
        "specs": {"material": "Fits Everybody (47% nylon, 33% viscose, 20% elastane)", "neckline": ["V-neck", "Square neck", "Scoop neck", "Thong", "Brief"], "available_sizes": "XXS-4X", "colorways": ["Bone", "Cocoa", "Marble", "Onyx", "Clay"]},
    },
    {
        "sku": "FASH-002",
        "name": "Skims Soft Lounge Long Slip Dress",
        "brand": "Skims",
        "category": "Fashion",
        "subcategory": "Dresses",
        "price": 88.00,
        "description": "The dress that's everywhere on Instagram. Skims' Soft Lounge fabric is buttery-soft and stretchy for an effortless, body-skimming look. Wear it out or lounge in it at home.",
        "specs": {"material": "Soft Lounge (56% nylon, 26% modal, 17% elastane, 1% spandex)", "length": "Maxi", "available_sizes": "XXS-4X", "colorways": ["Bone", "Onyx", "Cocoa", "Sienna"]},
    },
    {
        "sku": "FASH-003",
        "name": "Aritzia Super Puff Long Jacket",
        "brand": "Aritzia",
        "category": "Fashion",
        "subcategory": "Jackets",
        "price": 325.00,
        "description": "The iconic NYC winter coat. Aritzia's Super Puff in a longline silhouette with 700-fill-power responsibly sourced down. Warm, stylish, and instantly recognizable on every woman in Manhattan from November to March.",
        "specs": {"fill": "700-fill-power duck down", "shell": "Nylon (water-resistant)", "length": "Longline (hits mid-thigh)", "available_sizes": "2XS-3X", "colorways": ["Black", "Camel", "Ivory", "Burgundy", "Forest Green"]},
    },
    # --- Home ---
    {
        "sku": "HOME-001",
        "name": "Dyson V15 Detect",
        "brand": "Dyson",
        "category": "Home",
        "subcategory": "Vacuums",
        "price": 749.00,
        "description": "Dyson's most powerful cordless vacuum. Laser reveals invisible dust, intelligent suction auto-adjusts to floor type. HEPA filtration captures 99.97% of particles.",
        "specs": {"suction": "230 AW", "battery": "60 min runtime", "filtration": "HEPA (captures 0.3 microns)", "laser": "Green laser dust detection", "smart_display": True},
    },
    {
        "sku": "HOME-002",
        "name": "Instant Pot Duo 7-in-1",
        "brand": "Instant Pot",
        "category": "Home",
        "subcategory": "Appliances",
        "price": 99.00,
        "description": "The world's best-selling multi-cooker. Pressure cooker, slow cooker, rice cooker, steamer, sauté pan, yogurt maker, and warmer in one. 6-quart capacity.",
        "specs": {"capacity": "6 quarts", "functions": 7, "programs": 13, "material": "Stainless steel inner pot", "safety_features": 10},
    },
    {
        "sku": "HOME-003",
        "name": "YETI Rambler 30oz Tumbler",
        "brand": "YETI",
        "category": "Home",
        "subcategory": "Drinkware",
        "price": 45.00,
        "description": "YETI's iconic stainless steel tumbler with MagSlider lid. Double-wall vacuum insulation keeps drinks cold for hours and hot all morning. Dishwasher safe.",
        "specs": {"capacity": "30oz", "insulation": "Double-wall vacuum", "material": "18/8 stainless steel", "dishwasher_safe": True, "lid": "MagSlider", "colors": ["Black", "Charcoal", "White", "Navy", "Rescue Red"]},
    },
]

USERS = [
    {
        "name": "Alex Chen",
        "email": "alex@example.com",
        "preferences": {"sizes": {"clothing": "M", "shoes": "10"}, "brands": ["Apple", "Nike", "Patagonia"], "categories": ["Electronics", "Footwear"]},
    },
    {
        "name": "Jordan Smith",
        "email": "jordan@example.com",
        "preferences": {"sizes": {"clothing": "L", "shoes": "11"}, "brands": ["Sony", "Adidas", "Arc'teryx"], "categories": ["Electronics", "Clothing"]},
    },
    {
        "name": "Sam Rivera",
        "email": "sam@example.com",
        "preferences": {"sizes": {"clothing": "S", "shoes": "8"}, "brands": ["Allbirds", "Uniqlo", "YETI"], "categories": ["Footwear", "Home", "Clothing"]},
    },
    {
        "name": "Mia Hoffman",
        "email": "mia@example.com",
        "preferences": {"sizes": {"clothing": "XS", "shoes": "7.5", "activewear": "XS"}, "brands": ["Skims", "Alo Yoga", "Diptyque", "Rhode", "Aritzia"], "categories": ["Beauty & Wellness", "Activewear", "Fashion"]},
    },
]


def days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).isoformat()


def setup_sqlite() -> dict[str, int]:
    """Create tables and seed data. Returns mapping of sku -> product_id."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # Products
    product_ids: dict[str, int] = {}
    for p in PRODUCTS:
        cur = conn.execute(
            "INSERT OR IGNORE INTO products (sku, name, brand, category, subcategory, price, description, specs) VALUES (?,?,?,?,?,?,?,?)",
            (p["sku"], p["name"], p["brand"], p["category"], p["subcategory"], p["price"], p["description"], json.dumps(p["specs"])),
        )
        conn.execute(
            "INSERT OR REPLACE INTO inventory (product_id, quantity, warehouse, last_updated) VALUES ((SELECT id FROM products WHERE sku=?), ?, 'Seattle-WA', ?)",
            (p["sku"], 42, days_ago(0)),
        )
        product_ids[p["sku"]] = conn.execute("SELECT id FROM products WHERE sku=?", (p["sku"],)).fetchone()[0]

    # Price history — show some price drops to make "is this a good deal?" interesting
    price_history = [
        ("ELEC-003", [(399.00, 90), (379.00, 60), (349.00, 0)]),   # Sony headphones: $399 → $379 → $349
        ("ELEC-001", [(1799.00, 120), (1699.00, 60), (1599.00, 0)]),  # MacBook: $1799 → $1599
        ("FOOT-001", [(150.00, 90), (120.00, 45), (150.00, 0)]),    # Air Max: was on sale, back up
        ("HOME-001", [(799.00, 90), (749.00, 0)]),                   # Dyson: slight drop
        ("CLTH-003", [(799.00, 90), (799.00, 0)]),                   # Arc'teryx: no change
    ]
    for sku, history in price_history:
        pid = product_ids[sku]
        for price, days in history:
            conn.execute(
                "INSERT INTO price_history (product_id, price, recorded_at) VALUES (?,?,?)",
                (pid, price, days_ago(days)),
            )

    # Users
    user_ids: list[int] = []
    for u in USERS:
        conn.execute(
            "INSERT OR IGNORE INTO users (name, email, preferences) VALUES (?,?,?)",
            (u["name"], u["email"], json.dumps(u["preferences"])),
        )
        uid = conn.execute("SELECT id FROM users WHERE email=?", (u["email"],)).fetchone()[0]
        user_ids.append(uid)

    # Orders — varied statuses, ages, and items per user
    orders_data = [
        # Alex: delivered MacBook (3 weeks ago), processing AirPods (2 days ago)
        (user_ids[0], "delivered", 22, [("ELEC-001", 1, 1599.00)]),
        (user_ids[0], "processing", 2, [("ELEC-004", 1, 249.00)]),
        # Jordan: delivered Sony headphones (5 days ago), shipped Arc'teryx jacket (8 days ago)
        (user_ids[1], "delivered", 5, [("ELEC-003", 1, 349.00)]),
        (user_ids[1], "shipped", 8, [("CLTH-003", 1, 799.00)]),
        # Sam: delivered Allbirds + Uniqlo jacket (14 days ago), returned Dyson (35 days ago)
        (user_ids[2], "delivered", 14, [("FOOT-003", 1, 128.00), ("CLTH-004", 1, 89.00)]),
        (user_ids[2], "returned", 35, [("HOME-001", 1, 749.00)]),
        # Mia: delivered Alo leggings + Diptyque Baies (6 days ago), delivered Skims bodysuit (20 days ago), processing Aritzia Super Puff (1 day ago)
        (user_ids[3], "delivered", 6, [("ACTV-001", 1, 128.00), ("BEAU-001", 1, 75.00)]),
        (user_ids[3], "delivered", 20, [("FASH-001", 2, 62.00)]),
        (user_ids[3], "processing", 1, [("FASH-003", 1, 325.00)]),
    ]
    for user_id, status, days, items in orders_data:
        total = sum(price * qty for _, qty, price in items)
        cur = conn.execute(
            "INSERT INTO orders (user_id, status, created_at, total) VALUES (?,?,?,?)",
            (user_id, status, days_ago(days), total),
        )
        order_id = cur.lastrowid
        for sku, qty, price in items:
            conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, price_at_purchase) VALUES (?,?,?,?)",
                (order_id, product_ids[sku], qty, price),
            )

    # Reviews — mixed opinions to make "what do people think?" interesting
    reviews_data = [
        ("ELEC-003", user_ids[1], 5, "Best headphones I've ever owned. The ANC is absolutely incredible — blocks out everything on my commute. Sound quality is warm and detailed. Battery lasts forever. Worth every penny.", 10),
        ("ELEC-003", user_ids[0], 4, "Great headphones overall. ANC is top-notch and comfort is excellent for long sessions. My only gripe is they're a bit warm after 2+ hours. Would still recommend.", 7),
        ("ELEC-001", user_ids[0], 5, "The M4 MacBook Pro is shockingly fast. Final Cut renders that took 20 minutes on my old Intel Mac now take 3. The display is gorgeous. Battery life is genuinely all-day.", 4),
        ("FOOT-003", user_ids[2], 4, "Super comfortable and lightweight. Great for walking around the city. They do run slightly narrow — I'd suggest sizing up half a size if you have wide feet.", 3),
        ("CLTH-004", user_ids[2], 5, "Incredibly warm for such a light jacket. Packs into nothing. I wear this under my heavier shell in winter and it's become my most-used layer. Amazing value.", 5),
        ("HOME-001", user_ids[2], 3, "Powerful vacuum but the battery dies faster than advertised on thick carpet — more like 35 minutes than 60. The laser dust detection is a cool gimmick but I'd prioritize battery life.", 12),
        ("CLTH-003", user_ids[1], 5, "This jacket is bombproof. Wore it in a full Scottish winter — rain, sleet, wind — and stayed completely dry. Yes it's expensive, but you buy it once and it lasts a decade.", 2),
        ("ACTV-001", user_ids[3], 5, "These are THE legging. I own them in four colors. They don't roll down, they're squat-proof, and they look sleek enough to wear to brunch after Pilates. Worth every dollar.", 3),
        ("BEAU-001", user_ids[3], 5, "This candle is my whole personality. Every single person who comes to my apartment asks what smells so good. Baies is the only candle I'll ever burn.", 2),
        ("FASH-001", user_ids[3], 5, "I ordered two. The fabric is insane — so soft and stretchy. Looks flawless under everything. The V-neck in Bone is the most-worn item in my closet.", 15),
        ("BEAU-003", user_ids[3], 4, "It really does make your lips look like they have a treatment on them — plump and glossy. The Watermelon Glaze shade is so pretty. Only complaint is it disappears after eating.", 4),
        ("FASH-003", user_ids[0], 4, "Got this for my girlfriend and she absolutely loves it. The black longline is incredibly warm and looks amazing in the city. Runs slightly large — she sized down.", 8),
    ]
    for sku, user_id, rating, body, days in reviews_data:
        conn.execute(
            "INSERT INTO reviews (product_id, user_id, rating, body, created_at) VALUES (?,?,?,?,?)",
            (product_ids[sku], user_id, rating, body, days_ago(days)),
        )

    conn.commit()
    conn.close()
    print(f"SQLite database created at {DB_PATH}")
    return product_ids


def setup_chroma(product_ids: dict[str, int]) -> None:
    """Build Chroma vector store from product names + descriptions."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    docs = []
    for p in PRODUCTS:
        # Combine name, description, category, and key specs into one document
        text = (
            f"{p['name']} by {p['brand']}. "
            f"Category: {p['category']} > {p['subcategory']}. "
            f"Price: ${p['price']}. "
            f"{p['description']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "product_id": product_ids[p["sku"]],
                "sku": p["sku"],
                "name": p["name"],
                "brand": p["brand"],
                "category": p["category"],
                "subcategory": p["subcategory"],
                "price": p["price"],
            },
        ))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name="products",
    )
    print(f"Chroma index created at {CHROMA_PATH} ({len(docs)} products)")


if __name__ == "__main__":
    product_ids = setup_sqlite()
    setup_chroma(product_ids)
    print("Setup complete.")
