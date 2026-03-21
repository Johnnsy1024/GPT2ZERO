from typing import List

from tokenizer import GPT2Tokenizer


def _fallback_char_encoding(tokenizer, text: str) -> List[int]:
    return (
        [tokenizer.bos_token_id]
        + [ord(char) % 1000 + 10 for char in text]
        + [tokenizer.eos_token_id]
    )


def encode_training_text(tokenizer, text: str) -> List[int]:
    """Encode a training sample while preserving the existing GPT-2 fallback."""
    if isinstance(tokenizer, GPT2Tokenizer):
        try:
            return tokenizer.encode(text, add_special_tokens=True)
        except Exception:
            return _fallback_char_encoding(tokenizer, text)

    return tokenizer.encode(text, add_special_tokens=True)


def encode_prompt_text(tokenizer, prompt: str) -> List[int]:
    """Encode a prompt for generation while preserving the existing fallback."""
    if isinstance(tokenizer, GPT2Tokenizer):
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        except:
            token_ids = _fallback_char_encoding(tokenizer, prompt)
    else:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if not token_ids:
        bos_token_id = getattr(tokenizer, "bos_token_id", 1)
        token_ids = [bos_token_id]

    return token_ids


def decode_generated_ids(tokenizer, token_ids: List[int]) -> str:
    """Decode generated token ids while preserving the existing GPT-2 fallback."""
    if isinstance(tokenizer, GPT2Tokenizer):
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except:
            return "".join(chr((token_id - 10) % 256) for token_id in token_ids if token_id >= 10)

    return tokenizer.decode(token_ids, skip_special_tokens=True)
