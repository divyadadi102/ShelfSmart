from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.models import Stock
from app.database import get_db
from app.models import User
from app.services.stock import update_daily_sales_rate, apply_sales_to_stock
from datetime import datetime, timezone
router = APIRouter()
    
@router.get("/api/stock")
def get_stock_items(
    session: Session = Depends(get_db),
    # 🔴 Comment this out or remove it for now
    # current_user: User = Depends(get_current_user)
):
    try:
        stock_items = session.exec(
            select(Stock)
        ).all()
        return stock_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")

@router.put("/api/stock/update/{stock_id}")
def update_stock_quantity(stock_id: int, added_quantity: int, session: Session = Depends(get_db)):
    stock_item = session.get(Stock, stock_id)
    if not stock_item:
        raise HTTPException(status_code=404, detail="Stock item not found")

    stock_item.item_inventory += added_quantity
    stock_item.last_inventory_updated_at = datetime.now(timezone.utc)
    session.add(stock_item)
    session.commit()
    session.refresh(stock_item)
    return stock_item

@router.post("/api/stock/update-daily-sales")
def trigger_daily_sales_update(session: Session = Depends(get_db)):
    from app.services.stock import update_daily_sales_rate
    update_daily_sales_rate()
    return {"message": "Daily sales rate updated"}

@router.post("/api/stock/apply-sales")
def api_apply_sales(session: Session = Depends(get_db)):
    from app.services.stock import apply_sales_to_stock
    apply_sales_to_stock(session)
    return {"message": "Sales applied to stock."}