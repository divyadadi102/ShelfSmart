from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlmodel import Session
from typing import Optional

from ..database import get_db
from ..models import User
from ..core.security import get_current_user
from ..services.upload import process_upload_file

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload sales data files and perform forecasts for all periods (today/tomorrow/7 days).
    Automatically write to the Sales table, record the Upload table, perform forecasts, and return summary and chart data.
    """
    if file.content_type not in [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported.")

    try:
        result = await process_upload_file(
            file=file,
            user=current_user,
            session=session,
            prediction_type="today"  # 默认值，实际会生成所有三种预测
        )

        return {
            "message": "✅ All predictions (today, tomorrow, next week) completed.",
            "rows_inserted": result["rows_inserted"],
            "summary": result["prediction_summary"],
            "chart_data": result["chart_data"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: ❌ {str(e)}")
    

    