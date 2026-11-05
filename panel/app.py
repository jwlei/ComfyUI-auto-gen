"""The panel with all its pipelines on it."""
from panel import image, image_tools, server, shorts, video  # noqa: F401, each adds its side to the server

app = server.app
main = server.main
