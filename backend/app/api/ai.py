"""from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.services.ai_service import get_llm_response
from app.api.auth import get_current_user# depends on your JWT logic

router = APIRouter()

@router.post("/api/ai/query")
def query_ai(
    question: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    try:
        response = get_llm_response(question=question, user=user, db=db)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.auth import get_current_user
from app.models import User
from app.services.ai_service import get_llm_response
from app.database import get_db as get_session
from sqlmodel import Session

router = APIRouter()

class AIQuery(BaseModel):
    question: str

@router.post("/api/ai/query")
def ai_query(
    payload: AIQuery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)  # ✅ Inject the DB session
):
    response = get_llm_response(question=payload.question, user=current_user, db=db)
    return {"response": response}

