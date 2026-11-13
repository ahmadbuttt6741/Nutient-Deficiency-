"""
Authentication and User Management Module.
Handles SQLite user database, secure salted PBKDF2-SHA256 password hashing,
credential verification, user registration, and password recovery.
"""

import sqlite3
import hashlib
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

# Database path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "users.db"

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
    """
    Hashes a password using PBKDF2 with SHA-256 and a cryptographically secure salt.
    Returns (hex_digest, salt).
    """
    if salt is None:
        salt = os.urandom(32)
    # 100,000 iterations of PBKDF2-HMAC-SHA256
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return key.hex(), salt

def verify_password(stored_hash: str, salt: bytes, password_to_check: str) -> bool:
    """Verifies a plain-text password against the stored PBKDF2 hash and salt."""
    key = hashlib.pbkdf2_hmac("sha256", password_to_check.encode("utf-8"), salt, 100000)
    return key.hex() == stored_hash

def init_db():
    """Initializes the users table and seeds default demo accounts if not present."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt BLOB NOT NULL,
        role TEXT NOT NULL DEFAULT 'agronomist',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()

    # Ensure standard accounts with requested credentials exist and are up to date
    default_users = [
        ("botany", "botany@agrivision.ai", "Botany Department (Admin)", "nanstack", "admin"),
        ("admin", "admin@agrivision.ai", "Chief Agronomist (Admin)", "nanstack", "admin"),
        ("researcher", "researcher@nanstack.tech", "Nanstack AI Researcher", "nanstack", "researcher")
    ]
    
    now = datetime.utcnow().isoformat()
    for uname, email, fname, pwd, role in default_users:
        pwd_hash, salt = hash_password(pwd)
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", (uname.lower(), email.lower()))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                """
                UPDATE users SET password_hash = ?, salt = ?, full_name = ?, role = ?, is_active = 1
                WHERE id = ?
                """,
                (pwd_hash, salt, fname, role, row["id"])
            )
        else:
            cursor.execute(
                """
                INSERT INTO users (username, email, full_name, password_hash, salt, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (uname, email, fname, pwd_hash, salt, role, now)
            )
    conn.commit()
    
    conn.close()

def authenticate_user(identifier: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates a user by email or username.
    Returns (success, user_dict, message).
    """
    if not identifier or not password:
        return False, None, "Please provide both username/email and password."

    identifier = identifier.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
        (identifier, identifier)
    )
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return False, None, "Invalid credentials. Account not found."

    if not user_row["is_active"]:
        return False, None, "This account is inactive. Please contact support."

    if not verify_password(user_row["password_hash"], user_row["salt"], password):
        return False, None, "Invalid email or password. Please try again."

    user_data = {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"],
        "full_name": user_row["full_name"],
        "role": user_row["role"],
        "created_at": user_row["created_at"]
    }
    return True, user_data, "Login successful."

def register_user(username: str, email: str, full_name: str, password: str) -> Tuple[bool, str]:
    """
    Registers a new user with input validation.
    Returns (success, message).
    """
    username = username.strip().lower()
    email = email.strip().lower()
    full_name = full_name.strip()

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9_.-]+$", username):
        return False, "Username can only contain alphanumeric characters, dots, and underscores."
    
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, email):
        return False, "Please enter a valid email address."

    if not full_name or len(full_name) < 2:
        return False, "Please enter your full name."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_connection()
    cursor = conn.cursor()

    # Check for existing user
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already taken. Please choose another."

    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return False, "An account with this email already exists."

    pwd_hash, salt = hash_password(password)
    now = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT INTO users (username, email, full_name, password_hash, salt, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, 'agronomist', 1, ?)
        """,
        (username, email, full_name, pwd_hash, salt, now)
    )
    conn.commit()
    conn.close()
    return True, "Account registered successfully! You may now sign in."

def reset_password(email: str, new_password: str) -> Tuple[bool, str]:
    """
    Resets the password for a given email address.
    Returns (success, message).
    """
    email = email.strip().lower()
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters long."

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,))
    user_row = cursor.fetchone()
    if not user_row:
        conn.close()
        return False, "No account found matching this email address."

    pwd_hash, salt = hash_password(new_password)
    cursor.execute(
        "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
        (pwd_hash, salt, user_row["id"])
    )
    conn.commit()
    conn.close()
    return True, "Password has been successfully updated! You can now log in."

# Auto-initialize database on import
init_db()
