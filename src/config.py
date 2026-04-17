import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

wandb_token = os.getenv("WANDB_API_KEY")

if not wandb_token:
    raise ValueError("API keys are not set in environment variables")