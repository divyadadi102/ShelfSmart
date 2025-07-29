# app/services/stock.py
import datetime as dt
from sqlmodel import Session, select
from app.models import Product, Stock, Sales
from app.database import engine
from datetime import date
from datetime import date as dt_date
from typing import Optional

def populate_stock_from_product(session: Session):
    products = session.exec(select(Product)).all()
    
    for p in products:
        # check if stock already exists for this item in this store
        existing = session.exec(
            select(Stock)
            .where(Stock.item_nbr == p.item_nbr)
            .where(Stock.store_nbr == p.store_nbr)
        ).first()

        if not existing:
            # Add only if not present in stock
            new_stock = Stock(
                item_nbr=p.item_nbr,
                item_name=p.item_name,
                item_category=p.item_category,
                item_inventory=10,  # or user-defined
                store_nbr=p.store_nbr,
                date=p.date,
            )
            session.add(new_stock)
    
    session.commit()


def update_daily_sales_rate():
    """Recompute average daily sales for every item currently in stock."""
    with Session(engine) as session:
        item_nbrs = session.exec(select(Stock.item_nbr)).all()
        for item_nbr in item_nbrs:
            sales_data = session.exec(
                select(Sales.unit_sales, Sales.date).where(Sales.item_nbr == item_nbr)
            ).all()

            if not sales_data:
                continue

            total_units_sold = sum(float(s.unit_sales or 0) for s in sales_data)
            unique_days = {s.date for s in sales_data}
            if not unique_days:
                continue

            avg = total_units_sold / len(unique_days)

            for stock_row in session.exec(select(Stock).where(Stock.item_nbr == item_nbr)).all():
                stock_row.daily_sales_rate = round(avg, 1)
                session.add(stock_row)

        session.commit()

def apply_sales_to_stock(session: Session, upto_date: Optional[date] = None):
    sales_q = select(Sales)
    if upto_date:
        sales_q = sales_q.where(Sales.date <= upto_date)

    sales = session.exec(sales_q).all()

    for sale in sales:
        stock_entry = session.exec(
            select(Stock)
            .where(Stock.item_nbr == sale.item_nbr)
            .where(Stock.store_nbr == sale.store_nbr)
        ).first()

        if stock_entry:
            # 🔐 Only apply if sale is newer than last_applied
            if (stock_entry.last_sales_applied_date is None) or (sale.date > stock_entry.last_sales_applied_date):
                print(f"Applying sale for item {sale.item_nbr} on {sale.date} - units sold: {sale.unit_sales}")
                print(f"Before: {stock_entry.item_inventory}")
                stock_entry.item_inventory = max(0, stock_entry.item_inventory - int(sale.unit_sales or 0))
                stock_entry.last_sales_applied_date = sale.date
                print(f"After: {stock_entry.item_inventory}")

    session.commit()



