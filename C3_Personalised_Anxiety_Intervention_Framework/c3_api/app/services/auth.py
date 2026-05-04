"""C3 JWT authentication.

HS256, 24h expiry. Secret from C3_SECRET_KEY environment variable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY
from app.models.schemas import TokenData

logger = logging.getLogger("c3.auth")

# tokenUrl points to /v3/auth/login which is a stub — clients call /v3/register
# to obtain tokens in this Phase 3 implementation. Kept for OpenAPI docs.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v3/auth/login", auto_error=False)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Sign and return a JWT containing `data` plus `exp`."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> TokenData:
    """FastAPI dependency — decode JWT and return TokenData.

    In this Phase 3 build JWT is optional (auto_error=False) so smoke tests
    can run without needing a login flow. In production (Phase 7) this will
    be tightened to auto_error=True.
    """
    if token is None:
        # Allow unauthenticated requests — Phase 3 permissive mode.
        return TokenData(user_id=None)

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return TokenData(user_id=user_id)
    except JWTError as exc:
        logger.warning(f"JWT decode failed: {exc}")
        raise credentials_exception from exc
