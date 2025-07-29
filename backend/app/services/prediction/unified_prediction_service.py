#!/usr/bin/env python3
"""
unified_prediction_service.py

Unified prediction service that calls simple_predictor.py with different dates.
Supports: today, tomorrow, 7-day predictions
Now supports learning item names from user's CSV data!

Usage: 
    python unified_prediction_service.py <data_source> <prediction_type>
    python unified_prediction_service.py data.csv today
    python unified_prediction_service.py data.csv tomorrow  
    python unified_prediction_service.py data.csv 7days
"""

import sys
import os
import pandas as pd
import numpy as np
import pathlib
import json
import requests
from datetime import datetime, timedelta
from io import StringIO
from app.models import Forecast
from app.database import get_db
from typing import Optional
from sqlalchemy.dialects.postgresql import insert
from sqlmodel import Session, select
from app.models import Sales
from datetime import date, timedelta, datetime
from app.config import DEMO_DATE


# Import the main predictor and mapper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_predictor import LocalSalesPredictor
from item_mapper import ItemMapper

def predict_dataframe(
    self,
    df: pd.DataFrame,
    prediction_type: str = 'tomorrow',
    user_id: int = 1,
    source_file: Optional[str] = None,
    save_results: bool = True
):
    """Prediction methods that support DataFrame input (for API file upload calls)"""
    return self.predict(
        data_source=df,
        prediction_type=prediction_type,
        user_id=user_id,
        save_results=save_results
    )




def save_forecast_results(predictions: list[dict], user_id: int, model_version: str = "joblib", source_file: str = None):
    """保存预测结果到数据库 - 修复数据类型问题"""
    print(f"🔄 开始保存预测结果: {len(predictions)} 条记录")
    
    for_db = []
    uploaded_time = datetime.utcnow()

    for i, record in enumerate(predictions):
        try:
            # 数据类型转换和清理
            def safe_convert_to_int(value, default=None):
                """安全转换为整数"""
                if value is None:
                    return default
                if isinstance(value, str):
                    if value.strip() == '' or value.upper() in ['NAN', 'NULL', 'NONE']:
                        return default
                    try:
                        return int(float(value))  # 先转float再转int，处理"1.0"这种情况
                    except (ValueError, TypeError):
                        return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default

            def safe_convert_to_date(value, default=None):
                """安全转换为日期"""
                if value is None:
                    return default
                if isinstance(value, str):
                    # 如果是字符串，直接返回（PostgreSQL会自动处理）
                    return value
                return str(value)

            # 转换数据类型
            forecast = Forecast(
                user_id=user_id,
                store_nbr=safe_convert_to_int(record.get("store_nbr")),
                item_nbr=safe_convert_to_int(record.get("item_nbr")),
                prediction_date=safe_convert_to_date(record.get("prediction_date")),
                predicted_sales=float(record.get("predicted_sales", 0)),
                item_name=record.get("item_name"),
                category=record.get("category"),
                item_class=safe_convert_to_int(record.get("item_class")),  # 转换为整数
                perishable=bool(record.get("perishable", False)),
                model_version=model_version,
                uploaded_at=uploaded_time,
                source_file=source_file or "system_prediction"
            )
            for_db.append(forecast)
            
            # 每100条记录打印一次进度
            if (i + 1) % 100 == 0:
                print(f"   准备了 {i + 1}/{len(predictions)} 条记录")
                
        except Exception as e:
            print(f"❌ 处理第 {i+1} 条记录时出错: {e}")
            print(f"   问题记录: {record}")
            # 打印详细的字段信息用于调试
            print(f"   store_nbr: {record.get('store_nbr')} (type: {type(record.get('store_nbr'))})")
            print(f"   item_nbr: {record.get('item_nbr')} (type: {type(record.get('item_nbr'))})")
            print(f"   item_class: {record.get('item_class')} (type: {type(record.get('item_class'))})")
            continue

    print(f"📦 准备写入 {len(for_db)} 条记录到数据库")

    with next(get_db()) as session:
        try:
            # 转换为dict列表
            records_to_insert = []
            for f in for_db:
                try:
                    record_dict = f.dict(exclude={"id"})
                    records_to_insert.append(record_dict)
                except Exception as e:
                    print(f"⚠️ 跳过无效记录: {e}")
                    continue
            
            print(f"   成功转换 {len(records_to_insert)} 条记录")

            if not records_to_insert:
                print("❌ 没有有效记录可以插入")
                return

            # 去重主键，防止冲突
            import pandas as pd
            df_insert = pd.DataFrame(records_to_insert)
            before = len(df_insert)
            df_insert = df_insert.drop_duplicates(
                subset=["user_id", "store_nbr", "item_nbr", "prediction_date"], keep="last"
            )
            after = len(df_insert)
            if before != after:
                print(f"⚠️ 去重后记录数从 {before} 减少到 {after}")
            records_to_insert = df_insert.to_dict("records")

            # 插入数据库
            from sqlalchemy.dialects.postgresql import insert
            stmt = insert(Forecast).values(records_to_insert)

            update_fields = {
                "predicted_sales": stmt.excluded.predicted_sales,
                "item_name": stmt.excluded.item_name,
                "category": stmt.excluded.category,
                "item_class": stmt.excluded.item_class,
                "perishable": stmt.excluded.perishable,
                "model_version": stmt.excluded.model_version,
                "uploaded_at": stmt.excluded.uploaded_at,
                "source_file": stmt.excluded.source_file,
            }

            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "store_nbr", "item_nbr", "prediction_date"],
                set_=update_fields
            )

            session.execute(stmt)
            session.commit()
            
            print(f"✅ 成功写入 {len(records_to_insert)} 条预测记录到数据库")
            
        except Exception as e:
            import traceback
            print(f"❌ 数据库写入失败: {e}")
            traceback.print_exc()
            session.rollback()
            raise RuntimeError(f"Failed to upsert forecast records: {str(e)}")



    
class UnifiedPredictionService:
    def _get_user_store(self, db: Session, user_id: int) -> int:
        from app.models import User
        user = db.get(User, user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        return user.store_nbr
    
    def __init__(self, model_dir=None):
        if model_dir and os.path.exists(model_dir):
            print(f"✅ Using manually specified model directory: {model_dir}")
        else:
            print("🔍 Searching for model directory...")
            possible_paths = [
                "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/prediction/my_saved_model_resaved",
                "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/my_saved_model_resaved",
                "./my_saved_model_resaved",
                "../my_saved_model_resaved", 
                "./model",
                "../model"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_dir = path
                    print(f"📁 Found model directory: {model_dir}")
                    break

        if not model_dir or not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ Model directory not found: {model_dir}")

        self.predictor = LocalSalesPredictor(model_dir)
        self.mapper = ItemMapper()
        self.original_data = None
        
    def load_model(self):
        """Load the prediction model"""
        return self.predictor.load_model()
    
    # def predict(self, data_source, prediction_type='tomorrow', save_results=True, user_id: int = 1,
    #             db: Optional[Session] = None):
    #     """
    #     Make predictions for different time periods
        
    #     Args:
    #         data_source (str): Path to data file or URL
    #         prediction_type (str): 'today', 'tomorrow', or '7days'
    #         save_results (bool): Whether to save results to files
            
    #     Returns:
    #         dict: Prediction results with chart data
    #     """
    #     print(f"🚀 {prediction_type.title()} Sales Prediction")
    #     print("=" * 50)
                
    #     if isinstance(data_source, pd.DataFrame):
    #         df = data_source.copy()
    #         print(f"📊 Using provided DataFrame with {len(df)} records")
    #     elif db is not None:
            
    #         # 🛢️ By default, data from the last 60 days is retrieved from the database
    #         # today = date.today()
    #         # start_date = today - timedelta(days=60)

    #         # Get the maximum date from the sales table
    #         stmt_max_date = select(Sales.date).order_by(Sales.date.desc()).limit(1)
    #         max_date_row = db.exec(stmt_max_date).first()
    #         if not max_date_row:
    #             raise RuntimeError("The sales table has no data, so it is impossible to make predictions.")
    #         latest_date = max_date_row

    #         today = latest_date + timedelta(days=1)  # Use the latest date + 1 day as today
    #         start_date = today - timedelta(days=60)

    #         print(f"🛢️ Loading sales data from DB for user {user_id}, date range: {start_date} to {today}")
    #         # Step 1: Query sales table
    #         stmt_sales = select(Sales).where(
    #             Sales.store_nbr == self._get_user_store(db, user_id),
    #             Sales.date.between(start_date, today)
    #         )
    #         sales_rows = db.exec(stmt_sales).all()
    #         df_sales = pd.DataFrame([row.dict() for row in sales_rows])

    #         # Step 2: Query product table
    #         from app.models import Product
    #         stmt_product = select(Product.item_nbr, Product.item_name)
    #         product_rows = db.exec(stmt_product).all()
    #         df_product = pd.DataFrame(product_rows, columns=["item_nbr", "item_name"])

    #         # Step 3: Merge
    #         df = df_sales.merge(df_product, on="item_nbr", how="left")
    #         print("Data after merge:", df.head())

    #     else:
    #         raise ValueError("You must provide either a DataFrame or a DB session")
                
    #     print("DB columns:", df.columns.tolist())
    #     print(df.head())

    #     # 自动聚合去重，保留所有模型需要的字段
    #     agg_dict = {
    #         'unit_sales': 'sum',
    #         'onpromotion': 'first',
    #         'category': 'first',
    #         'item_class': 'first',
    #         'perishable': 'first',
    #         'price': 'first',
    #         'cost_price': 'first',
    #         'item_name': 'first'
    #     }
    #     agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    #     if df.duplicated(['store_nbr', 'item_nbr', 'date']).any():
    #         print("⚠️ 检测到重复的 (store_nbr, item_nbr, date) 销售记录，自动聚合")
    #         df = df.groupby(['store_nbr', 'item_nbr', 'date'], as_index=False).agg(agg_dict)

    #     # Store original data for item mapping learning
    #     self.original_data = df.to_dict('records')
        
    #     # Check if user data contains item names
    #     has_item_names = 'item_name' in df.columns
    #     if has_item_names:
    #         unique_items = df['item_nbr'].nunique()
    #         items_with_names = df.dropna(subset=['item_name'])['item_nbr'].nunique()
    #         print(f"📦 Found item names in data: {items_with_names}/{unique_items} items have names")
            
    #         # Let mapper learn from the data
    #         self.mapper.learn_from_data(self.original_data)
    #     else:
    #         print("📦 No item_name column found, will use default item mappings")
        
    #     # Debug: Check data characteristics
    #     print(f"\n🔍 Data Analysis:")
    #     print(f"   Unique items: {df['item_nbr'].nunique()}")
    #     print(f"   Unique stores: {df['store_nbr'].nunique()}")
    #     print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
    #     # Sample a few records to see the data
    #     print(f"\n📋 Sample data:")
    #     sample_items = df[['store_nbr', 'item_nbr', 'item_name', 'unit_sales']].head(5)
    #     for i, row in sample_items.iterrows():
    #         print(f"   Store {row['store_nbr']}, Item {row['item_nbr']}: {row.get('item_name', 'No name')} - Sales: {row['unit_sales']}")
        
    #     # Determine prediction dates based on type
    #     prediction_dates = self._get_prediction_dates(prediction_type)
        
    #     all_predictions = []
        
    #     for i, (date_obj, date_str, day_info) in enumerate(prediction_dates):
    #         print(f"🔮 {day_info}: {date_str}")
            
    #         try:
    #             # Use simple_predictor to make prediction (no duplicate logic)
    #             daily_prediction = self.predictor.predict(
    #                 input_data=df,
    #                 prediction_date=date_str,
    #                 save_result=False
    #             )
                
    #             if daily_prediction is not None and len(daily_prediction) > 0:
    #                 # Debug: Check prediction variety
    #                 pred_values = daily_prediction['predicted_sales'].values
    #                 unique_preds = len(set(pred_values))
    #                 print(f"   🔍 Generated {len(pred_values)} predictions with {unique_preds} unique values")
    #                 print(f"   📊 Range: {pred_values.min():.3f} - {pred_values.max():.3f}")
                    
    #                 # Sample a few predictions
    #                 if len(pred_values) >= 5:
    #                     sample_idx = [0, len(pred_values)//4, len(pred_values)//2, 3*len(pred_values)//4, len(pred_values)-1]
    #                     sample_preds = [pred_values[i] for i in sample_idx]
    #                     print(f"   🎯 Sample predictions: {[f'{x:.3f}' for x in sample_preds]}")
                    
    #                 # Add day information for multi-day predictions
    #                 if prediction_type == '7days':
    #                     daily_prediction = daily_prediction.copy()
    #                     daily_prediction['day_number'] = i + 1
    #                     daily_prediction['day_name'] = date_obj.strftime('%A')
    #                     daily_prediction['date_full'] = date_str
                    
    #                 all_predictions.append(daily_prediction)
    #                 print(f"   ✅ Generated {len(daily_prediction)} predictions")
    #             else:
    #                 print(f"   ⚠️ No predictions generated for {date_str}")
                    
    #         except Exception as e:
    #             print(f"   ❌ Failed to predict for {date_str}: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             continue
        
    #     if not all_predictions:
    #         raise RuntimeError("No predictions generated for any date")
        
    #     # Combine predictions
    #     if len(all_predictions) == 1:
    #         combined_predictions = all_predictions[0]
    #     else:
    #         combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
    #     # Get item information and merge with predictions
    #     item_info = df[['item_nbr', 'category', 'item_class', 'perishable']].drop_duplicates()
        
    #     # Include item names if available in original data
    #     if has_item_names:
    #         item_names = df[['item_nbr', 'item_name']].dropna().drop_duplicates()
    #         item_info = item_info.merge(item_names, on='item_nbr', how='left')
        
    #     # Merge with item information
    #     detailed_predictions = combined_predictions.merge(item_info, on='item_nbr', how='left')
        
    #     # Enrich with readable names using mapper (pass original data for learning)
    #     enriched_predictions = self.mapper.enrich_predictions(
    #         detailed_predictions.to_dict('records'),
    #         original_data=self.original_data
    #     )
    #     enriched_df = pd.DataFrame(enriched_predictions)
        
    #     # Generate analysis
    #     analysis = self._generate_analysis(enriched_df, prediction_type)
        
    #     # Create chart data
    #     chart_data = self._create_chart_data(enriched_predictions, prediction_type)
        
    #     # Prepare results
    #     results = {
    #         'prediction_type': prediction_type,
    #         'prediction_dates': [date_str for _, date_str, _ in prediction_dates],
    #         'data_info': {
    #             'total_records': len(df),
    #             'unique_items': df['item_nbr'].nunique(),
    #             'unique_stores': df['store_nbr'].nunique(),
    #             'has_item_names': has_item_names,
    #             'date_range': {
    #                 'start': df['date'].min(),
    #                 'end': df['date'].max()
    #             }
    #         },
    #         'summary': analysis,
    #         'detailed_predictions': enriched_predictions,
    #         'chart_data': chart_data,
    #         'generated_at': datetime.now().isoformat()
    #     }
        
    #     # Save results if requested
    #     if save_results:
    #         self._save_results(results, enriched_df, prediction_type)

    #     # 写入数据库
    #     save_forecast_results(
    #         predictions=enriched_df.to_dict("records"),
    #         user_id=1,
    #         model_version=self.predictor.load_method,
    #         source_file=data_source if isinstance(data_source, str) else None
    #     )

    #     return results
    
    def predict(self, data_source, prediction_type='tomorrow', save_results=True, user_id: int = 1):
        """
        Make predictions for different time periods
        
        Args:
            data_source (str): Path to data file or URL
            prediction_type (str): 'today', 'tomorrow', or '7days'
            save_results (bool): Whether to save results to files
            
        Returns:
            dict: Prediction results with chart data
        """
        print(f"🚀 {prediction_type.title()} Sales Prediction")
        print("=" * 50)
        
        # 数据库写入标志
        should_write_to_db = True
                
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            print(f"📊 Using provided DataFrame with {len(df)} records")
        elif data_source is None:
            
            # 🛢️ 使用内部 session 从数据库加载数据
            with next(get_db()) as db:
                # Get the maximum date from the sales table
                stmt_max_date = select(Sales.date).order_by(Sales.date.desc()).limit(1)
                max_date_row = db.exec(stmt_max_date).first()
                if not max_date_row:
                    raise RuntimeError("The sales table has no data, so it is impossible to make predictions.")
                latest_date = max_date_row

                today = latest_date + timedelta(days=1)  # Use the latest date + 1 day as today
                start_date = today - timedelta(days=60)

                print(f"🛢️ Loading sales data from DB for user {user_id}, date range: {start_date} to {today}")
                # Step 1: Query sales table
                stmt_sales = select(Sales).where(
                    Sales.store_nbr == self._get_user_store(db, user_id),
                    Sales.date.between(start_date, today)
                )
                sales_rows = db.exec(stmt_sales).all()
                df_sales = pd.DataFrame([row.dict() for row in sales_rows])

                # Step 2: Query product table
                from app.models import Product
                stmt_product = select(Product.item_nbr, Product.item_name)
                product_rows = db.exec(stmt_product).all()
                df_product = pd.DataFrame(product_rows, columns=["item_nbr", "item_name"])

                # Step 3: Merge
                df = df_sales.merge(df_product, on="item_nbr", how="left")
                print("Data after merge:", df.head())

        else:
            # 文件模式 - 需要创建新的数据库会话用于写入
            if data_source.startswith('http'):
                df = self._load_from_url(data_source)
            else:
                df = pd.read_csv(data_source)
                print(f"📊 Loaded {len(df)} records from file: {data_source}")
            
            # 文件模式也使用内部 session，不需要额外创建
            print("��️ 文件模式将使用内部数据库会话")
            
        print("DB columns:", df.columns.tolist())
        print(df.head())

        # 自动聚合去重，保留所有模型需要的字段
        agg_dict = {
            'unit_sales': 'sum',
            'onpromotion': 'first',
            'category': 'first',
            'item_class': 'first',
            'perishable': 'first',
            'price': 'first',
            'cost_price': 'first',
            'item_name': 'first'
        }
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        if df.duplicated(['store_nbr', 'item_nbr', 'date']).any():
            print("⚠️ 检测到重复的 (store_nbr, item_nbr, date) 销售记录，自动聚合")
            df = df.groupby(['store_nbr', 'item_nbr', 'date'], as_index=False).agg(agg_dict)

        # Store original data for item mapping learning
        self.original_data = df.to_dict('records')
        
        # Check if user data contains item names
        has_item_names = 'item_name' in df.columns
        if has_item_names:
            unique_items = df['item_nbr'].nunique()
            items_with_names = df.dropna(subset=['item_name'])['item_nbr'].nunique()
            print(f"📦 Found item names in data: {items_with_names}/{unique_items} items have names")
            
            # Let mapper learn from the data
            self.mapper.learn_from_data(self.original_data)
        else:
            print("📦 No item_name column found, will use default item mappings")
        
        # Debug: Check data characteristics
        print(f"\n🔍 Data Analysis:")
        print(f"   Unique items: {df['item_nbr'].nunique()}")
        print(f"   Unique stores: {df['store_nbr'].nunique()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Sample a few records to see the data
        print(f"\n📋 Sample data:")
        sample_items = df[['store_nbr', 'item_nbr', 'item_name', 'unit_sales']].head(5)
        for i, row in sample_items.iterrows():
            print(f"   Store {row['store_nbr']}, Item {row['item_nbr']}: {row.get('item_name', 'No name')} - Sales: {row['unit_sales']}")
        
        # Determine prediction dates based on type
        prediction_dates = self._get_prediction_dates(prediction_type)
        
        all_predictions = []
        
        for i, (date_obj, date_str, day_info) in enumerate(prediction_dates):
            print(f"🔮 {day_info}: {date_str}")
            
            try:
                # Use simple_predictor to make prediction (no duplicate logic)
                daily_prediction = self.predictor.predict(
                    input_data=df,
                    prediction_date=date_str,
                    save_result=False
                )
                
                if daily_prediction is not None and len(daily_prediction) > 0:
                    # Debug: Check prediction variety
                    pred_values = daily_prediction['predicted_sales'].values
                    unique_preds = len(set(pred_values))
                    print(f"   🔍 Generated {len(pred_values)} predictions with {unique_preds} unique values")
                    print(f"   📊 Range: {pred_values.min():.3f} - {pred_values.max():.3f}")
                    
                    # Sample a few predictions
                    if len(pred_values) >= 5:
                        sample_idx = [0, len(pred_values)//4, len(pred_values)//2, 3*len(pred_values)//4, len(pred_values)-1]
                        sample_preds = [pred_values[i] for i in sample_idx]
                        print(f"   🎯 Sample predictions: {[f'{x:.3f}' for x in sample_preds]}")
                    
                    # Add day information for multi-day predictions
                    if prediction_type == '7days':
                        daily_prediction = daily_prediction.copy()
                        daily_prediction['day_number'] = i + 1
                        daily_prediction['day_name'] = date_obj.strftime('%A')
                        daily_prediction['date_full'] = date_str
                    
                    all_predictions.append(daily_prediction)
                    print(f"   ✅ Generated {len(daily_prediction)} predictions")
                else:
                    print(f"   ⚠️ No predictions generated for {date_str}")
                    
            except Exception as e:
                print(f"   ❌ Failed to predict for {date_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_predictions:
            raise RuntimeError("No predictions generated for any date")
        
        # Combine predictions
        if len(all_predictions) == 1:
            combined_predictions = all_predictions[0]
        else:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Get item information and merge with predictions
        item_info = df[['item_nbr', 'category', 'item_class', 'perishable']].drop_duplicates()
        
        # Include item names if available in original data
        if has_item_names:
            item_names = df[['item_nbr', 'item_name']].dropna().drop_duplicates()
            item_info = item_info.merge(item_names, on='item_nbr', how='left')
        
        # Merge with item information
        detailed_predictions = combined_predictions.merge(item_info, on='item_nbr', how='left')
        
        # Enrich with readable names using mapper (pass original data for learning)
        enriched_predictions = self.mapper.enrich_predictions(
            detailed_predictions.to_dict('records'),
            original_data=self.original_data
        )
        enriched_df = pd.DataFrame(enriched_predictions)
        
        # Generate analysis
        analysis = self._generate_analysis(enriched_df, prediction_type)
        
        # Create chart data
        chart_data = self._create_chart_data(enriched_predictions, prediction_type)
        
        # Prepare results
        results = {
            'prediction_type': prediction_type,
            'prediction_dates': [date_str for _, date_str, _ in prediction_dates],
            'data_info': {
                'total_records': len(df),
                'unique_items': df['item_nbr'].nunique(),
                'unique_stores': df['store_nbr'].nunique(),
                'has_item_names': has_item_names,
                'date_range': {
                    'start': df['date'].min(),
                    'end': df['date'].max()
                }
            },
            'summary': analysis,
            'detailed_predictions': enriched_predictions,
            'chart_data': chart_data,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save results if requested
        if save_results:
            self._save_results(results, enriched_df, prediction_type)

        # 改进的数据库写入逻辑 - 使用内部 session
        if should_write_to_db:
            try:
                print(f"💾 开始写入预测结果到数据库...")
                
                # 使用内部的 save_forecast_results 函数
                save_forecast_results(
                    predictions=enriched_df.to_dict("records"),
                    user_id=user_id,
                    model_version=self.predictor.load_method,
                    source_file=None
                )
                print(f"✅ 成功写入 {len(enriched_df)} 条预测记录到数据库")
                
            except Exception as e:
                print(f"❌ 数据库写入失败: {e}")
                import traceback
                traceback.print_exc()
                # 不抛出异常，让预测结果仍然返回
        else:
            print("⚠️ 跳过数据库写入（文件模式或无数据库连接）")

        return results


    # 新增：支持传入会话的数据库写入函数
    def save_forecast_results_with_session(predictions: list[dict], user_id: int, model_version: str = "joblib", 
                                        source_file: str = None, session: Session = None):
        """保存预测结果到数据库 - 支持传入会话"""
        print(f"🔄 开始保存预测结果: {len(predictions)} 条记录")
        
        if not session:
            # 如果没有传入会话，使用原来的逻辑
            return save_forecast_results(predictions, user_id, model_version, source_file)
        
        for_db = []
        uploaded_time = datetime.utcnow()

        for i, record in enumerate(predictions):
            try:
                # 数据类型转换和清理
                def safe_convert_to_int(value, default=None):
                    """安全转换为整数"""
                    if value is None:
                        return default
                    if isinstance(value, str):
                        if value.strip() == '' or value.upper() in ['NAN', 'NULL', 'NONE']:
                            return default
                        try:
                            return int(float(value))  # 先转float再转int，处理"1.0"这种情况
                        except (ValueError, TypeError):
                            return default
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                def safe_convert_to_date(value, default=None):
                    """安全转换为日期"""
                    if value is None:
                        return default
                    if isinstance(value, str):
                        # 如果是字符串，直接返回（PostgreSQL会自动处理）
                        return value
                    return str(value)

                # 转换数据类型
                forecast = Forecast(
                    user_id=user_id,
                    store_nbr=safe_convert_to_int(record.get("store_nbr")),
                    item_nbr=safe_convert_to_int(record.get("item_nbr")),
                    prediction_date=safe_convert_to_date(record.get("prediction_date")),
                    predicted_sales=float(record.get("predicted_sales", 0)),
                    item_name=record.get("item_name"),
                    category=record.get("category"),
                    item_class=safe_convert_to_int(record.get("item_class")),  # 转换为整数
                    perishable=bool(record.get("perishable", False)),
                    model_version=model_version,
                    uploaded_at=uploaded_time,
                    source_file=source_file or "system_prediction"
                )
                for_db.append(forecast)
                
                # 每100条记录打印一次进度
                if (i + 1) % 100 == 0:
                    print(f"   准备了 {i + 1}/{len(predictions)} 条记录")
                    
            except Exception as e:
                print(f"❌ 处理第 {i+1} 条记录时出错: {e}")
                continue

        print(f"📦 准备写入 {len(for_db)} 条记录到数据库")

        try:
            # 转换为dict列表
            records_to_insert = []
            for f in for_db:
                try:
                    record_dict = f.dict(exclude={"id"})
                    records_to_insert.append(record_dict)
                except Exception as e:
                    print(f"⚠️ 跳过无效记录: {e}")
                    continue
            
            print(f"   成功转换 {len(records_to_insert)} 条记录")

            if not records_to_insert:
                print("❌ 没有有效记录可以插入")
                return

            # 去重主键，防止冲突
            import pandas as pd
            df_insert = pd.DataFrame(records_to_insert)
            before = len(df_insert)
            df_insert = df_insert.drop_duplicates(
                subset=["user_id", "store_nbr", "item_nbr", "prediction_date"], keep="last"
            )
            after = len(df_insert)
            if before != after:
                print(f"⚠️ 去重后记录数从 {before} 减少到 {after}")
            records_to_insert = df_insert.to_dict("records")

            # 插入数据库
            from sqlalchemy.dialects.postgresql import insert
            stmt = insert(Forecast).values(records_to_insert)

            update_fields = {
                "predicted_sales": stmt.excluded.predicted_sales,
                "item_name": stmt.excluded.item_name,
                "category": stmt.excluded.category,
                "item_class": stmt.excluded.item_class,
                "perishable": stmt.excluded.perishable,
                "model_version": stmt.excluded.model_version,
                "uploaded_at": stmt.excluded.uploaded_at,
                "source_file": stmt.excluded.source_file,
            }

            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "store_nbr", "item_nbr", "prediction_date"],
                set_=update_fields
            )

            session.execute(stmt)
            session.commit()
            
            print(f"✅ 成功写入 {len(records_to_insert)} 条预测记录到数据库")
            
        except Exception as e:
            import traceback
            print(f"❌ 数据库写入失败: {e}")
            traceback.print_exc()
            session.rollback()
            raise RuntimeError(f"Failed to upsert forecast records: {str(e)}")
    
    def _get_prediction_dates(self, prediction_type):
        """Get prediction dates based on type"""

        #  Demo date for testing purposes
        # today = datetime.now()
        today = datetime(2017, 8, 16)
        #today = datetime.combine(DEMO_DATE, datetime.min.time())
        
        if prediction_type == 'today':
            return [(today, today.strftime('%Y-%m-%d'), "Today")]
            
        elif prediction_type == 'tomorrow':
            tomorrow = today + timedelta(days=1)
            return [(tomorrow, tomorrow.strftime('%Y-%m-%d'), "Tomorrow")]
            
        elif prediction_type == '7days':
            dates = []
            for i in range(7):
                date_obj = today + timedelta(days=i+1)
                date_str = date_obj.strftime('%Y-%m-%d')
                day_name = date_obj.strftime('%A')
                day_info = f"Day {i+1}/7 ({day_name})"
                dates.append((date_obj, date_str, day_info))
            return dates
        
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def _load_from_url(self, url):
        """Load data from URL"""
        print(f"🌐 Loading data from URL: {url}")
        
        # Handle Google Drive URLs
        if 'drive.google.com' in url and '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            print(f"   🔗 Converted Google Drive URL: {file_id}")
        
        # Handle Dropbox URLs
        elif 'dropbox.com' in url and 'dl=0' in url:
            url = url.replace('dl=0', 'dl=1')
            print(f"   🔗 Converted Dropbox URL for direct download")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; SalesPredictor/1.0)'}
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Check if response looks like CSV
            content_type = response.headers.get('content-type', '').lower()
            if 'text/csv' not in content_type and 'application/csv' not in content_type:
                # Try to detect if it's CSV by checking first few lines
                first_lines = response.text[:1000]
                if ',' not in first_lines and '\t' not in first_lines:
                    print(f"⚠️ Warning: Content doesn't appear to be CSV format")
                    print(f"   Content-Type: {content_type}")
            
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            print(f"   ✅ Successfully loaded {len(df)} records from URL")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download from URL: {e}")
            raise
        except pd.errors.EmptyDataError:
            print(f"❌ Downloaded file appears to be empty")
            raise
        except pd.errors.ParserError as e:
            print(f"❌ Failed to parse CSV data: {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error loading from URL: {e}")
            raise
    
    def _generate_analysis(self, df, prediction_type):
        """Generate analysis based on prediction type"""
        base_stats = {
            'total_predictions': int(len(df)),
            'total_items': int(df['item_nbr'].nunique()),
            'total_stores': int(df['store_nbr'].nunique()),
            'total_predicted_sales': float(df['predicted_sales'].sum()),
            'average_sales_per_prediction': float(df['predicted_sales'].mean()),
            'max_prediction': float(df['predicted_sales'].max()),
            'min_prediction': float(df['predicted_sales'].min())
        }
        
        # Top predictions with item names (use actual item names if available)
        item_name_col = 'item_name' if 'item_name' in df.columns else 'item_nbr'
        
        # Debug: Check prediction distribution before sorting
        print(f"\n🔍 Debug - Prediction Analysis:")
        print(f"   Unique prediction values: {df['predicted_sales'].nunique()}")
        print(f"   Max prediction value: {df['predicted_sales'].max():.3f}")
        print(f"   How many records have max value: {(df['predicted_sales'] == df['predicted_sales'].max()).sum()}")
        
        # Sample some high and low predictions for debugging
        sorted_predictions = df.sort_values('predicted_sales', ascending=False)
        print(f"   Top 5 predictions by value:")
        for i, (idx, row) in enumerate(sorted_predictions.head(5).iterrows()):
            print(f"     {i+1}. Store {row['store_nbr']}, Item {row['item_nbr']}: {row['predicted_sales']:.3f}")
        
        print(f"   Bottom 5 predictions by value:")
        for i, (idx, row) in enumerate(sorted_predictions.tail(5).iterrows()):
            print(f"     {i+1}. Store {row['store_nbr']}, Item {row['item_nbr']}: {row['predicted_sales']:.3f}")
        
        # Get diverse top predictions (mix of high values from different stores/items)
        top_predictions_diverse = []
        
        # First, get some of the highest predictions
        highest_predictions = sorted_predictions.head(5).to_dict('records')
        for pred in highest_predictions:
            if item_name_col == 'item_nbr':
                pred['item_name'] = f"Item #{pred['item_nbr']}"
            top_predictions_diverse.append(pred)
        
        # Then, get one top prediction per store (to show diversity)
        store_top_predictions = df.groupby('store_nbr').apply(
            lambda x: x.loc[x['predicted_sales'].idxmax()]
        ).reset_index(drop=True)
        
        # Add diverse store predictions (excluding already added ones)
        for _, pred in store_top_predictions.head(10).iterrows():
            pred_dict = pred.to_dict()
            if item_name_col == 'item_nbr':
                pred_dict['item_name'] = f"Item #{pred_dict['item_nbr']}"
            
            # Check if this store is already represented
            if not any(p['store_nbr'] == pred_dict['store_nbr'] for p in top_predictions_diverse[:5]):
                top_predictions_diverse.append(pred_dict)
                if len(top_predictions_diverse) >= 1000:
                    break
        
        # Fill remaining slots with diverse item predictions
        item_top_predictions = df.groupby('item_nbr').apply(
            lambda x: x.loc[x['predicted_sales'].idxmax()]
        ).reset_index(drop=True).sort_values('predicted_sales', ascending=False)
        
        for _, pred in item_top_predictions.iterrows():
            pred_dict = pred.to_dict()
            if item_name_col == 'item_nbr':
                pred_dict['item_name'] = f"Item #{pred_dict['item_nbr']}"
            
            # Check if this item is already represented
            if not any(p['item_nbr'] == pred_dict['item_nbr'] for p in top_predictions_diverse):
                top_predictions_diverse.append(pred_dict)
                if len(top_predictions_diverse) >= 1000:
                    break
        
        # Store performance
        store_performance = df.groupby('store_nbr').agg({
            'predicted_sales': ['sum', 'mean', 'count']
        }).round(2)
        store_performance.columns = ['total_sales', 'avg_sales', 'item_count']
        store_performance = store_performance.reset_index().sort_values('total_sales', ascending=False)
        
        # Category performance
        category_performance = df.groupby('category_name').agg({
            'predicted_sales': ['sum', 'mean', 'count']
        }).round(2)
        category_performance.columns = ['total_sales', 'avg_sales', 'item_count']
        category_performance = category_performance.reset_index().sort_values('total_sales', ascending=False)
        
        analysis = {
            **base_stats,
            'top_predictions': top_predictions_diverse,
            'store_performance': store_performance.to_dict('records'),
            'category_performance': category_performance.to_dict('records')
        }
        
        # Add day-by-day analysis for 7-day predictions
        if prediction_type == '7days' and 'day_number' in df.columns:
            daily_summary = df.groupby(['day_number', 'day_name', 'date_full']).agg({
                'predicted_sales': ['sum', 'mean', 'count']
            }).round(2)
            daily_summary.columns = ['total_sales', 'avg_sales', 'predictions_count']
            daily_summary = daily_summary.reset_index()
            
            analysis['daily_breakdown'] = daily_summary.to_dict('records')
            analysis['peak_day'] = daily_summary.loc[daily_summary['total_sales'].idxmax(), 'day_name']
            analysis['peak_day_sales'] = float(daily_summary['total_sales'].max())
        
        return analysis
    
    def _create_chart_data(self, predictions, prediction_type):
        """Create chart data for frontend"""
        chart_data_category = self.mapper.create_chart_data(predictions, 'category', 10)
        chart_data_items = self.mapper.create_chart_data(predictions, 'item', 1000)
        
        chart_data = {
            'by_category': chart_data_category,
            'by_item': chart_data_items
        }
        
        # Add daily chart for 7-day predictions
        if prediction_type == '7days':
            df = pd.DataFrame(predictions)
            if 'day_name' in df.columns:
                daily_chart = df.groupby('day_name')['predicted_sales'].sum().sort_index()
                chart_data['by_day'] = {
                    'title': 'Daily Sales Predictions (7 Days)',
                    'labels': daily_chart.index.tolist(),
                    'values': daily_chart.values.tolist(),
                    'colors': self.mapper._generate_colors(len(daily_chart)),
                    'total': float(daily_chart.sum()),
                    'group_by': 'day'
                }
        
        return chart_data
    
    def _save_results(self, results, df, prediction_type):
        """Save results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        # Set up result directory
        result_dir = pathlib.Path(__file__).parent / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed predictions
        csv_file = result_dir / f"{prediction_type}_predictions_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Detailed predictions saved: {csv_file}")

        # Save JSON summary
        json_file = result_dir / f"{prediction_type}_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 JSON summary saved: {json_file}")

        # Save learned item mappings
        if self.original_data and any('item_name' in record for record in self.original_data):
            mappings_file = result_dir / f"learned_mappings_{timestamp}.json"
            self.mapper.save_mappings(str(mappings_file))
            print(f"💾 Learned item mappings saved: {mappings_file}")

def find_model_directory():
    """Helper function to find model directory"""
    print("🔍 Searching for model directory...")
    
    possible_paths = [
        "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/prediction/my_saved_model_resaved",
        "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/my_saved_model_resaved",
        "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/prediction/my_saved_model",
        "/Users/wangjiamin/Documents/GitHub/ShelfSmart/backend/app/services/my_saved_model",
        "./my_saved_model_resaved",
        "../my_saved_model_resaved",
        "./my_saved_model", 
        "../my_saved_model",
        "./model",
        "../model"
    ]
    
    for i, path in enumerate(possible_paths, 1):
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"   {i:2d}. {path} - {status}")
        
        if exists:
            # Check if it contains model files
            try:
                files = os.listdir(path)
                model_files = [f for f in files if f.endswith(('.pkl', '.txt', '.json'))]
                print(f"       📁 Contains {len(model_files)} model files: {model_files[:3]}{'...' if len(model_files) > 3 else ''}")
                return path
            except Exception as e:
                print(f"       ⚠️ Cannot read directory: {e}")
    
    print("❌ No valid model directory found!")
    return None



def main():
    print("🚀 Running main()")
    """Main function with command line interface - 支持数据库模式"""
    
    # 支持两种模式：
    # 1. python script.py <prediction_type> (数据库模式)
    # 2. python script.py <file_path> <prediction_type> (文件模式)
    
    if len(sys.argv) == 2:
        # 数据库模式
        prediction_type = sys.argv[1].lower()
        use_database = True
        data_source = None
        user_id = 1
        
    elif len(sys.argv) == 3:
        # 检查第一个参数是否是文件路径
        first_arg = sys.argv[1]
        if os.path.exists(first_arg) or first_arg.startswith('http'):
            # 文件模式
            data_source = first_arg
            prediction_type = sys.argv[2].lower()
            use_database = False
            user_id = 1
        else:
            print("❌ 第一个参数不是有效的文件路径或URL")
            sys.exit(1)
    else:
        print("Usage:")
        print("  数据库模式: python unified_prediction_service.py <prediction_type>")
        print("  文件模式:   python unified_prediction_service.py <file_path> <prediction_type>")
        print("\nPrediction types:")
        print("  today     - Predict for today")
        print("  tomorrow  - Predict for tomorrow") 
        print("  7days     - Predict for next 7 days")
        print("\nExamples:")
        print("  python unified_prediction_service.py today")
        print("  python unified_prediction_service.py 7days")
        print("  python unified_prediction_service.py data.csv today")
        sys.exit(1)
    
    if prediction_type not in ['today', 'tomorrow', '7days']:
        print(f"❌ Invalid prediction type: {prediction_type}")
        sys.exit(1)
    
    try:
        # Check if model directory exists
        print("🔍 Checking model directory...")
        model_dir = str(pathlib.Path(__file__).resolve().parents[2] / "models" / "my_saved_model_resaved")
        print(f"✅ Computed absolute model path: {model_dir}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"❌ Model directory does not exist: {model_dir}")

        service = UnifiedPredictionService(model_dir=model_dir)
        
        # Load model
        if not service.load_model():
            print("❌ Failed to load prediction model")
            sys.exit(1)
        
        # Make prediction
        if use_database:
            print(f"🛢️ Using database mode for user_id: {user_id}")
            # 不再需要手动管理数据库会话，predict() 内部会处理
            results = service.predict(
                data_source=None, 
                prediction_type=prediction_type, 
                user_id=user_id,
                save_results=True
            )
        else:
            print(f"📁 Using file mode: {data_source}")
            results = service.predict(data_source, prediction_type, user_id=user_id, save_results=True)
        
        # Print summary
        print(f"\n✅ {prediction_type.title()} prediction completed successfully!")
        print(f"📊 Generated {results['summary']['total_predictions']} predictions")
        print(f"💰 Total predicted sales: {results['summary']['total_predicted_sales']:.2f}")
        
        if prediction_type == '7days':
            print(f"🔥 Peak day: {results['summary']['peak_day']} ({results['summary']['peak_day_sales']:.2f})")
        
        # 显示写入数据库的情况
        print(f"\n💾 数据库写入状态:")
        print(f"   预测类型: {prediction_type}")
        print(f"   预测日期: {', '.join(results['prediction_dates'])}")
        
        # Show detailed item predictions with debugging
        print(f"\n🛍️ Top Selling Items Predictions:")
        print("=" * 100)
        top_items = results['summary']['top_predictions'][:10]  # Show top 10
        
        for i, item in enumerate(top_items, 1):
            store = item['store_nbr']
            item_name = item.get('item_name', f"Item #{item['item_nbr']}")
            category = item.get('category_name', 'Unknown Category')
            sales = item['predicted_sales']
            item_nbr = item['item_nbr']
            print(f"{i:2d}. Store {store:2d} | Item #{item_nbr:<6} | {item_name:<35} | {category:<20} | {sales:>6.2f} units")
        
        return 0
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
if __name__ == "__main__":
    sys.exit(main())
