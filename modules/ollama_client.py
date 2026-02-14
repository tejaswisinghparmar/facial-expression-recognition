"""
Ollama Integration Module
Sends ASL‑recognised text to a local Ollama instance for spelling/grammar correction.
"""

import threading
import ollama


# Default model — change to "mistral" if you prefer
DEFAULT_MODEL = "gemma3:1b"


def process_with_ollama(
    text: str,
    callback=None,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """
    Send *text* (raw ASL letter buffer) to a local Ollama model and ask it
    to correct spelling / grammar.

    Parameters
    ----------
    text : str
        The raw string assembled from ASL finger‑spelling.
    callback : callable, optional
        ``callback(corrected_text: str)`` — called on success.
        If *None*, the function blocks and returns the result.
    model : str
        Ollama model name (default ``llama3``).

    Returns
    -------
    str or None
        Corrected text when *callback* is None; otherwise None
        (result delivered via callback).
    """
    prompt = (
        "The following text was assembled from individual ASL finger‑spelled "
        "letters and may contain misspellings or missing spaces. "
        "Please correct the spelling and grammar, and return ONLY the "
        "corrected sentence — no explanations.\n\n"
        f"Raw text: {text}"
    )

    def _call():
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            corrected = response["message"]["content"].strip()
        except Exception as exc:
            corrected = f"[Ollama error] {exc}"

        if callback is not None:
            callback(corrected)
        return corrected

    if callback is not None:
        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        return None
    else:
        return _call()
