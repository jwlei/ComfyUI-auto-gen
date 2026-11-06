"""Run the installed OpenH3-IR node pack's compiler in ComfyUI's Python."""
import contextlib
import hashlib
import json
import os
import sys
import types


def split_qwen_reasoning(content):
    if "</think>" in content:
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    else:
        reasoning = ""
    if "<think>" in content or "</think>" in content or not content.strip():
        raise ValueError("Qwen returned incomplete reasoning or an empty answer.")
    return reasoning, content.strip()


def main():
    request = json.load(sys.stdin)
    root = request.pop("comfy_root")
    model_id = request.pop("model_id")
    sys.path.insert(0, root)
    package = types.ModuleType("autogen_openh3")
    package.__path__ = [os.path.join(root, "custom_nodes", "ComfyUI-OpenH3-IR")]
    sys.modules[package.__name__] = package
    with contextlib.redirect_stdout(sys.stderr):
        from autogen_openh3.compiler import compile_here
        from autogen_openh3.h3ir_client import build_payload

        if model_id == "qwen3-14b":
            import h3ir.backend as backend

            class QwenBackend(backend.Backend):
                def chat(self, *args, **kwargs):
                    reply = super().chat(*args, **kwargs)
                    try:
                        reasoning, reply.content = split_qwen_reasoning(reply.content)
                    except ValueError as e:
                        raise backend.BackendError(str(e)) from e
                    if reasoning:
                        reply.reasoning = "\n".join(filter(None, (reply.reasoning, reasoning)))
                    return reply

            backend.Backend = QwenBackend

        payload = build_payload(**request["brief"])
        def sha_of(path):
            with open(path, "rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest()

        result = compile_here(payload, sha_of=sha_of, llm_url=request["url"],
                              llm_model=request["model"], timeout=request["timeout"])
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
