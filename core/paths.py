"""Where everything lives. Each pipeline keeps its own runs and videos; logs and workflows are shared."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Moved only by the tests, so a real pipeline process can run against a throwaway folder.
DATA_ROOT = os.getenv("AUTOGEN_DATA") or PROJECT_ROOT
LOGS_DIR = os.path.join(DATA_ROOT, "logs")
WORKFLOWS_DIR = os.path.join(PROJECT_ROOT, "workflows")
# The workflows saved in ComfyUI, as the pipelines run them.
SAVED_WORKFLOWS_DIR = os.path.join(LOGS_DIR, "workflows")
# ComfyUI only sends progress to the client that queued the job, so the panel listens as this one.
COMFYUI_CLIENT_ID = "comfyui-auto-gen"


def comfyui_client_id():
    return os.getenv("COMFYUI_CLIENT_ID") or COMFYUI_CLIENT_ID
