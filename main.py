from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Simple FastAPI Server",
    description="A simple FastAPI server with basic endpoints",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    quantity: int = 1


class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    quantity: int
    total: float


items_db: dict[int, Item] = {}
item_counter = 0


@app.get("/")
def root():
    return {"message": "Welcome to the Simple FastAPI Server!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/items")
def get_items():
    return {
        "items": [
            {"id": id, **item.model_dump(), "total": item.price * item.quantity}
            for id, item in items_db.items()
        ]
    }


@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        return {"error": "Item not found"}
    item = items_db[item_id]
    return ItemResponse(
        id=item_id,
        name=item.name,
        description=item.description,
        price=item.price,
        quantity=item.quantity,
        total=item.price * item.quantity
    )


@app.post("/items")
def create_item(item: Item):
    global item_counter
    item_counter += 1
    items_db[item_counter] = item
    return ItemResponse(
        id=item_counter,
        name=item.name,
        description=item.description,
        price=item.price,
        quantity=item.quantity,
        total=item.price * item.quantity
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
