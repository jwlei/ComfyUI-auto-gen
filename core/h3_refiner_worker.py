"""Run the installed OpenH3-IR node pack's compiler in ComfyUI's Python."""
import contextlib
import hashlib
import json
import os
import sys
import types


def main():
    request = json.load(sys.stdin)
    root = request.pop("comfy_root")
    sys.path.insert(0, root)
    package = types.ModuleType("autogen_openh3")
    package.__path__ = [os.path.join(root, "custom_nodes", "ComfyUI-OpenH3-IR")]
    sys.modules[package.__name__] = package
    with contextlib.redirect_stdout(sys.stderr):
        from autogen_openh3.compiler import compile_here
        from autogen_openh3.h3ir_client import build_payload

        payload = build_payload(**request["brief"])
        def sha_of(path):
            with open(path, "rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest()

        result = compile_here(payload, sha_of=sha_of, llm_url=request["url"],
                              llm_model=request["model"], timeout=request["timeout"])
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
