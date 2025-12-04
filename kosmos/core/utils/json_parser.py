"""
Robust JSON parsing utility for LLM responses.

Handles various formats that models (especially local models like Ollama)
may produce, including:
- Direct JSON
- JSON wrapped in markdown code blocks
- JSON with trailing commas
- JSON with single quotes
- Mixed text/JSON responses
"""

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """Exception raised when JSON parsing fails after all strategies."""

    def __init__(self, message: str, original_text: str, attempts: int = 0):
        self.message = message
        self.original_text = original_text
        self.attempts = attempts
        super().__init__(f"{message} (tried {attempts} strategies)")


def parse_json_response(
    response_text: str, schema: dict[str, Any] | None = None, strict: bool = False
) -> dict[str, Any]:
    """
    Parse JSON from model response with multiple fallback strategies.

    Strategies tried in order:
    1. Direct JSON parse
    2. Extract from ```json code blocks (closed)
    2b. Extract from ```json code blocks (unclosed/truncated)
    3. Extract from ``` code blocks
    4. Extract JSON object using regex
    5. Clean common issues (trailing commas, single quotes)

    Args:
        response_text: Raw response text from the model
        schema: Optional expected schema (for validation/hints)
        strict: If True, only try direct parse (no fallbacks)

    Returns:
        Dict[str, Any]: Parsed JSON object

    Raises:
        JSONParseError: If all parsing strategies fail
    """
    if not response_text or not response_text.strip():
        raise JSONParseError("Empty response", response_text or "", 0)

    text = response_text.strip()
    attempts = 0

    # Strategy 1: Direct parse
    attempts += 1
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if strict:
            raise JSONParseError(
                f"JSON decode failed: {text[:100]}...", response_text, attempts
            ) from e

    # Strategy 2: Extract from ```json code blocks
    attempts += 1
    json_block_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2b: Extract from unclosed ```json blocks (truncated responses)
    attempts += 1
    unclosed_json_match = re.search(r"```json\s*([\s\S]+)", text)
    if unclosed_json_match and not json_block_match:
        # Try to find complete JSON object within the unclosed block
        block_content = unclosed_json_match.group(1).strip()
        json_obj_in_block = re.search(r"(\{[\s\S]*\})", block_content)
        if json_obj_in_block:
            try:
                return json.loads(json_obj_in_block.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: Extract from ``` code blocks (without json marker)
    attempts += 1
    code_block_match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        # Skip if it looks like code (has common code markers)
        if not any(
            marker in block_content for marker in ["def ", "class ", "import ", "function "]
        ):
            try:
                return json.loads(block_content)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Extract JSON object using regex (find first {...})
    attempts += 1
    json_obj_match = re.search(r"\{[\s\S]*\}", text)
    if json_obj_match:
        try:
            return json.loads(json_obj_match.group(0))
        except json.JSONDecodeError:
            # Try with cleaning
            pass

    # Strategy 5: Clean common issues and retry
    attempts += 1
    cleaned = _clean_json_string(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 5b: Try cleaning extracted JSON object
    if json_obj_match:
        attempts += 1
        cleaned_obj = _clean_json_string(json_obj_match.group(0))
        try:
            return json.loads(cleaned_obj)
        except json.JSONDecodeError:
            pass

    # Strategy 5c: Try cleaning code block content
    if json_block_match:
        attempts += 1
        cleaned_block = _clean_json_string(json_block_match.group(1).strip())
        try:
            return json.loads(cleaned_block)
        except json.JSONDecodeError:
            pass

    # All strategies failed
    logger.warning(f"JSON parsing failed after {attempts} attempts")
    logger.debug(f"Original text: {text[:500]}...")

    raise JSONParseError("Could not parse JSON from response", response_text, attempts)


def _clean_json_string(text: str) -> str:
    """
    Clean common JSON formatting issues.

    Handles:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Unquoted keys
    - Extra whitespace

    Args:
        text: JSON-like string to clean

    Returns:
        Cleaned JSON string
    """
    if not text:
        return text

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace single quotes with double quotes (careful with apostrophes)
    # Only replace if it looks like JSON quotes (around keys/values)
    text = re.sub(r"(?<=[{,\[:])\s*'([^']*?)'\s*(?=[,}\]:])", r'"\1"', text)
    text = re.sub(r"(?<=[{,\[])\s*'([^']*?)'\s*(?=:)", r'"\1"', text)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Remove any control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text


def extract_json_value(text: str, key: str) -> str | None:
    """
    Extract a specific key's value from potentially malformed JSON.

    Useful as a last resort when full JSON parsing fails.

    Args:
        text: Response text
        key: Key to extract

    Returns:
        Value if found, None otherwise
    """
    patterns = [
        rf'"{key}"\s*:\s*"([^"]*)"',  # "key": "value"
        rf'"{key}"\s*:\s*(\d+\.?\d*)',  # "key": number
        rf'"{key}"\s*:\s*(true|false)',  # "key": boolean
        rf'{key}\s*:\s*"([^"]*)"',  # key: "value" (unquoted key)
        rf"{key}\s*:\s*([^\n,}}]+)",  # key: value (loose)
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
