from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select
from ..database import get_db
from .. import models, schemas
from ..core.security import hash_password, verify_password, create_access_token
from ..schemas import LoginRequest
from ..models import PasswordResetToken, User
from ..core.security import get_current_user
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.core.config import settings
from datetime import datetime, timedelta

import random
import secrets

router = APIRouter()

conf = ConnectionConfig(
    MAIL_USERNAME = settings.MAIL_USERNAME,
    MAIL_PASSWORD = settings.MAIL_PASSWORD,
    MAIL_FROM = settings.MAIL_FROM,
    MAIL_PORT = settings.MAIL_PORT,
    MAIL_SERVER = settings.MAIL_SERVER,
    MAIL_STARTTLS = str(settings.MAIL_STARTTLS).lower() in ("1", "true", "yes", "on"),
    MAIL_SSL_TLS = str(settings.MAIL_SSL_TLS).lower() in ("1", "true", "yes", "on"),
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True
)

@router.post("/register", response_model=schemas.UserRead)
def register(user: schemas.UserCreate, session: Session = Depends(get_db)):
    db_user = session.exec(select(models.User).where(models.User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    random_store_nbr = random.randint(100000000, 999999999)
    new_user = models.User(
        name=user.name,
        email=user.email,
        hashed_password=hash_password(user.password),
        store_nbr=random_store_nbr,
        business_name=user.business_name
        # store_nbr = 1
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

@router.post("/login", response_model=schemas.Token)
def login(user: LoginRequest, session: Session = Depends(get_db)):
    db_user = session.exec(select(models.User).where(models.User.email == user.email)).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/user/me", response_model=schemas.UserRead)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@router.post("/forgot-password", status_code=200)
def forgot_password(req: schemas.ForgotPasswordRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_db)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user:
        return {"msg": "If your email exists, a reset link has been sent."}

    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=30)

    old_tokens = session.exec(
        select(PasswordResetToken).where(PasswordResetToken.user_id == user.id)
    ).all()
    for t in old_tokens:
        session.delete(t)

    prt = PasswordResetToken(user_id=user.id, token=token, expires_at=expires)
    session.add(prt)
    session.commit()

    reset_link = f"http://localhost:8080/reset-password?token={token}"
    email_body = f"""
    Hello,

    Please click the link below to reset your password:
    {reset_link}

    If you did not request this, please ignore this email.
    """

    message = MessageSchema(
        subject="ShelfSmart Password Reset",
        recipients=[user.email],
        body=email_body,
        subtype="plain"
    )

    fm = FastMail(conf)
    # send it as a background task so as not to block the main request
    background_tasks.add_task(fm.send_message, message)

    return {"msg": "If your email exists, a reset link has been sent."}


# submit new password
@router.post("/reset-password", status_code=200)
def reset_password(req: schemas.ResetPasswordRequest, session: Session = Depends(get_db)):
    token_obj = session.exec(
        select(PasswordResetToken).where(PasswordResetToken.token == req.token)
    ).first()
    if not token_obj or token_obj.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token invalid or expired")

    user = session.exec(select(User).where(User.id == token_obj.user_id)).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    user.hashed_password = hash_password(req.new_password)
    session.add(user)
    session.delete(token_obj)
    session.commit()
    return {"msg": "Password has been reset successfully"}
