import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URI, uuidRepresentation="standard")


def get_db():
    return client[os.getenv("MONGODB_DB", "curso_api")]
