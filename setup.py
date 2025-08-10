import os
import subprocess

# ===== Step 1: Install required packages =====
required_packages = [
    #"transformers",
    #"torch",
    "cohere",
    "python-dotenv",
    "pandas",
    "numpy",
    "requests"
]

print("[INFO] Installing required packages...")
for package in required_packages:
    subprocess.check_call(["pip", "install", package, "--quiet"])

# ===== Step 2: Import to verify installation =====
print("[INFO] Verifying imports...")
try:
    import transformers
    import torch
    import cohere
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    import requests
except ImportError as e:
    print(f"[ERROR] Missing module after install: {e}")
    exit(1)

# ===== Step 3: Create .env file =====
env_content = """# API Keys and Environment Variables

# Hugging Face API Token
HF_API_KEY='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'

# Cohere API Key
COHERE_API_KEY='mc4FkwQcGSV1QyJonOq1rEBOPZzeovm6uNfeBWHl'

# Add more keys if needed:
# YOUTUBE_API_KEY='AIzaSyBx3CCgsqrwsVBEMeoQ4fcZA1jzue12SQ4'
"""

env_path = ".env"

if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"[INFO] Created {env_path} file. Please fill in your API keys.")
else:
    print(f"[INFO] {env_path} already exists. Skipping creation.")

# ===== Step 4: Test .env loading =====
load_dotenv()
print("[INFO] Environment variables loaded:")
print("HF_API_KEY =", os.getenv("HF_API_KEY"))
print("COHERE_API_KEY =", os.getenv("COHERE_API_KEY"))

print("\n Setup complete")
