from pathlib import Path
import os

# ============================================================================
# LLM Backend Selection
# ============================================================================
USE_OLLAMA = True
USE_OPENAI = False

# ============================================================================
# OpenAI Configuration (optional)
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "API_KEY")
KEY_OWNER = os.getenv("KEY_OWNER", "NAME")

# ============================================================================
# Ollama Configuration (local model)
# ============================================================================
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.85"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.92"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "2000"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ============================================================================
# General Settings
# ============================================================================
DEBUG = False
MAX_CHUNK_SIZE = 4

# Model version to use
LLM_VERS = "ollama" if USE_OLLAMA else "gpt-4o-mini"

# Paths
BASE_DIR = f"{Path(__file__).resolve().parent.parent}"
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
