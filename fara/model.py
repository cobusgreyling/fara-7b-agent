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

    @staticmethod
    def _image_to_data_url(image_path: Path | str) -> str:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"

    def step(
        self,
        task: str,
        screenshot_path: Path | str,
        history: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run one inference turn.

        history is a list of prior assistant messages (thought + tool_call
        strings) and the user screenshot turns that preceded them, in the
        OpenAI chat-completion format. Returns the raw assistant text.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(screenshot_path)},
                    },
                    {"type": "text", "text": task},
                ],
            }
        )

        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"]
