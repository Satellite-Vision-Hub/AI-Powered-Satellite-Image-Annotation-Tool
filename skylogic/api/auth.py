"""
SkyLogic MAS — JWT Authentication
Sign-in, Sign-up, and token management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from skylogic.config import settings
from skylogic.database import get_db
from skylogic.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# --- Schemas ---

class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


# --- Utilities ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.jwt_expiry_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return {}


# --- Dependencies ---

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Get current user from JWT token. Returns None if no token."""
    if not token:
        return None

    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        return None

    user = db.query(User).filter(User.username == username).first()
    return user


async def require_auth(
    user: Optional[User] = Depends(get_current_user),
) -> User:
    """Require authentication — raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# --- Seed Default User ---

def seed_default_user(db: Session):
    """Seed the default 'SkyLogic' user on startup."""
    existing = db.query(User).filter(User.username == settings.seed_username).first()
    if existing:
        logger.info(f"Default user '{settings.seed_username}' already exists")
        return

    user = User(
        username=settings.seed_username,
        hashed_password=hash_password(settings.seed_password),
        full_name="SkyLogic Admin",
        is_active=True,
        is_admin=True,
    )
    db.add(user)
    db.commit()
    logger.info(f"Seeded default user: {settings.seed_username}")


# --- Routes ---

@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    user = User(
        username=user_data.username,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login and receive JWT token."""
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(require_auth)):
    """Get current authenticated user info."""
    return user
