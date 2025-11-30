# src/agent.py -- Day 9: E-commerce Voice Agent (Amazon India Style)

import os
import re
import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

logger = logging.getLogger("day9_amazon")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ------------------------------
# SIMPLE AMAZON-INDIA STYLE CATALOG
# ------------------------------

CATALOG = [
    {
        "id": "ab-mug-001",
        "name": "AmazonBasics Chai Mug",
        "description": "Simple stoneware mug for your chai or coffee.",
        "price": 299,
        "currency": "INR",
        "category": "mug",
    },
    {
        "id": "ab-tee-001",
        "name": "Amazon Brand Cotton T shirt",
        "description": "Soft cotton tee for daily wear.",
        "price": 399,
        "currency": "INR",
        "category": "tshirt",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "ab-coffee-001",
        "name": "Amazon Pantry Instant Coffee",
        "description": "Instant coffee powder 200g.",
        "price": 249,
        "currency": "INR",
        "category": "grocery",
    },
    {
        "id": "ab-bottle-001",
        "name": "AmazonBasics Steel Bottle 1L",
        "description": "Stainless steel insulated water bottle.",
        "price": 599,
        "currency": "INR",
        "category": "bottle",
    },
    {
        "id": "ab-phone-001",
        "name": "Amazon Budget Smartphone",
        "description": "Simple Android smartphone for calling and apps.",
        "price": 12000,
        "currency": "INR",
        "category": "mobile",
    },
]

ORDERS_FILE = "orders.json"
if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump([], f)

# ------------------------------
# HELPERS
# ------------------------------

def _load_orders():
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def _save_order(order):
    orders = _load_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

def find_product_by_id(pid: str):
    for p in CATALOG:
        if p["id"].lower() == pid.lower():
            return p
    return None

def find_product_by_ref(ref: str, candidates=None):
    if not ref:
        return None

    ref = ref.lower().strip()
    cand = candidates if candidates else CATALOG

    # numeric index reference ("1", "2")
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(cand):
            return cand[idx]

    # match ID directly
    for p in cand:
        if p["id"].lower() == ref:
            return p

    # name match
    for p in cand:
        if ref in p["name"].lower():
            return p

    return None

# ------------------------------
# Pydantic models
# ------------------------------

class CatalogFilter(BaseModel):
    category: Optional[str] = None
    q: Optional[str] = None
    max_price: Optional[int] = None


# ------------------------------
# User Session Data
# ------------------------------

@dataclass
class Userdata:
    cart: List[Dict[str, Any]] = field(default_factory=list)
    last_browse: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)


# ------------------------------
# TOOLS
# ------------------------------

@function_tool
async def show_catalog(ctx: RunContext[Userdata], filters: CatalogFilter = None) -> str:
    f = filters.dict() if filters else {}
    results = []

    for p in CATALOG:
        ok = True
        if f.get("category") and f["category"].lower() not in p["category"]:
            ok = False
        if f.get("max_price") and p["price"] > f["max_price"]:
            ok = False
        if f.get("q") and f["q"].lower() not in p["name"].lower():
            ok = False
        if ok:
            results.append(p)

    ctx.userdata.last_browse = results

    if not results:
        return "I could not find anything matching your search."

    msg = ["Here are some options:"]
    for idx, p in enumerate(results[:5], start=1):
        msg.append(f"{idx}. {p['name']} — {p['price']} INR (id: {p['id']})")

    msg.append("You can say 'add item 1 to my cart'. What would you like to do next?")
    return "\n".join(msg)


@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: str,
    quantity: int = 1,
    size: Optional[str] = None,
) -> str:
    candidates = ctx.userdata.last_browse or CATALOG
    prod = find_product_by_ref(product_ref, candidates) or find_product_by_ref(product_ref, CATALOG)

    if not prod:
        return "I couldn't find that product. Try using an item number or ID."

    ctx.userdata.cart.append(
        {"product_id": prod["id"], "quantity": quantity, "size": size}
    )

    return f"Added {quantity} × {prod['name']} to your cart. What would you like next?"


@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    cart = ctx.userdata.cart
    if not cart:
        return "Your cart is empty right now."

    total = 0
    lines = ["Here is your cart:"]
    for idx, item in enumerate(cart, start=1):
        p = find_product_by_id(item["product_id"])
        if not p:
            continue
        line_total = p["price"] * item["quantity"]
        total += line_total
        lines.append(f"{idx}. {p['name']} ×{item['quantity']} — {line_total} INR")

    lines.append(f"Total = {total} INR.")
    lines.append("You can say 'remove item 1' or 'place my order'. What would you like to do?")
    return "\n".join(lines)


@function_tool
async def remove_from_cart(ctx: RunContext[Userdata], product_ref: str) -> str:
    cart = ctx.userdata.cart
    if not cart:
        return "Your cart is empty already."

    candidates = []
    for item in cart:
        p = find_product_by_id(item["product_id"])
        if p:
            candidates.append(p)

    prod = find_product_by_ref(product_ref, candidates)
    if not prod:
        return "I could not find that item in your cart."

    product_id = prod["id"]
    new_cart = [c for c in cart if c["product_id"] != product_id]

    ctx.userdata.cart = new_cart

    if not new_cart:
        return f"I removed {prod['name']}. Your cart is empty now."

    return f"I removed {prod['name']} from your cart. Would you like to remove anything else?"


@function_tool
async def clear_cart(ctx: RunContext[Userdata]) -> str:
    ctx.userdata.cart = []
    return "I cleared your cart. What would you like to do next?"


@function_tool
async def place_order(ctx: RunContext[Userdata]) -> str:
    cart = ctx.userdata.cart
    if not cart:
        return "Your cart is empty. Add something first."

    line_items = []
    total = 0
    for item in cart:
        p = find_product_by_id(item["product_id"])
        if not p:
            continue
        total += p["price"] * item["quantity"]
        line_items.append(
            {
                "product_id": p["id"],
                "name": p["name"],
                "quantity": item["quantity"],
                "price": p["price"],
            }
        )

    order = {
        "id": f"order-{uuid.uuid4().hex[:8]}",
        "items": line_items,
        "total": total,
        "created_at": datetime.utcnow().isoformat(),
    }

    _save_order(order)
    ctx.userdata.orders.append(order)
    ctx.userdata.cart = []

    return f"Your mock Amazon-style order is placed! Order ID {order['id']}. Total: {total} INR. What next?"


@function_tool
async def last_order(ctx: RunContext[Userdata]) -> str:
    orders = ctx.userdata.orders or _load_orders()
    if not orders:
        return "You don't have any previous orders."

    ordc = orders[-1]
    msg = [f"Last order {ordc['id']}:"]
    for it in ordc["items"]:
        msg.append(f"- {it['name']} ×{it['quantity']} — {it['price']} INR")

    msg.append(f"Total: {ordc['total']} INR.")
    return "\n".join(msg)


# ------------------------------
# AGENT
# ------------------------------

class EcomAgent(Agent):
    def __init__(self):
        instructions = """
You are Aisha, a friendly Amazon India–style shopping assistant.
Use the provided tools. Do not make up products.
Always end with a short question.
"""
        super().__init__(
            instructions=instructions,
            tools=[
                show_catalog,
                add_to_cart,
                show_cart,
                remove_from_cart,
                clear_cart,
                place_order,
                last_order,
            ],
        )


# ------------------------------
# ENTRYPOINT
# ------------------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except:
        pass


async def entrypoint(ctx: JobContext):
    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya",
            style="Conversation",
            text_pacing=True,
        ),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=EcomAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
