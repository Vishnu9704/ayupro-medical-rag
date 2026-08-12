import os

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import bcrypt
from pydantic import BaseModel, EmailStr

load_dotenv()

app = FastAPI(title="AyuPro API")

# Allow requests from the Next.js frontend running on localhost.
# TODO: add your production frontend URL here once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor,
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- Request/response shapes -------------------------------------------
# Field names here match what the frontend signup/login forms already send.

class SignupRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    phone: str | None = None
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# --- Signup ---------------------------------------------------------------

@app.post("/api/signup")
def signup(payload: SignupRequest):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM users WHERE email = %s", (payload.email,)
            )
            if cursor.fetchone():
                raise HTTPException(
                    status_code=400,
                    detail="An account with this email already exists.",
                )

            password_hash = bcrypt.hashpw(
                payload.password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            cursor.execute(
                """
                INSERT INTO users (email, password_hash, phone_number, status)
                VALUES (%s, %s, %s, 'active')
                """,
                (payload.email, password_hash, payload.phone),
            )
            user_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO user_profiles (user_id, first_name, last_name)
                VALUES (%s, %s, %s)
                """,
                (user_id, payload.firstName, payload.lastName),
            )

        conn.commit()
        return {
            "id": user_id,
            "email": payload.email,
            "firstName": payload.firstName,
            "lastName": payload.lastName,
        }
    finally:
        conn.close()


# --- Login ------------------------------------------------------------

@app.post("/api/login")
def login(payload: LoginRequest):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT u.id, u.email, u.password_hash,
                       p.first_name, p.last_name
                FROM users u
                JOIN user_profiles p ON p.user_id = u.id
                WHERE u.email = %s
                """,
                (payload.email,),
            )
            user = cursor.fetchone()

        password_matches = user and bcrypt.checkpw(
            payload.password.encode("utf-8"),
            user["password_hash"].encode("utf-8"),
        )
        if not password_matches:
            raise HTTPException(
                status_code=401, detail="Invalid email or password."
            )

        # TODO: issue a real session token / JWT here instead of returning
        # raw user info. This is enough to prove the DB connection works.
        return {
            "id": user["id"],
            "email": user["email"],
            "firstName": user["first_name"],
            "lastName": user["last_name"],
        }
    finally:
        conn.close()