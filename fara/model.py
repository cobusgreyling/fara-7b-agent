"""Fara-7B model loader and single-turn inference."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

from .prompt import SYSTEM_PROMPT

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "model.gguf"
DEFAULT_MMPROJ = Path(__file__).resolve().parent.parent / "models" / "mmproj.gguf"


def image_to_data_url(image_path: Path | str) -> str:
    """Encode a PNG file as a base64 data URL for the chat handler."""
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


class FaraModel:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        mmproj_path: Path | str = DEFAULT_MMPROJ,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=verbose)
        self.llm = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def step(
        self,
        task: str,
        screenshot_path: Path | str,
        history: list[dict[str, Any]] | None = None,
        notes: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Run one inference turn.

        `history` is the message list from prior turns in OpenAI chat-completion
        format. `notes` are facts the model previously memorised via
        `pause_and_memorize_fact` — re-injected into the user text so the model
        does not lose them when older screenshots fall out of the sliding window.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)

        user_text = task
        if notes:
            bullets = "\n".join(f"- {n}" for n in notes)
            user_text = f"{task}\n\nMemorised facts:\n{bullets}"

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(screenshot_path)},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        )

        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"]
