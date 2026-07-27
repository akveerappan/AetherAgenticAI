#!/usr/bin/env python3
"""
Gemini client adapter for AetherFit.
Builds a Gemini client with API key rotation and redundancy support.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

try:
    from google import generativeai as genai
except ImportError:
    genai = None


def _get_all_api_keys() -> list:
    """
    Return all available Gemini API keys for rotation.
    Checks Streamlit secrets (GEMINI_API_KEY_1..4) then environment variables
    (GEMINI_API_KEY, GEMINI_API_KEY_1..4).
    """
    keys = []
    try:
        import streamlit as st

        for i in range(1, 5):
            try:
                key = st.secrets.get(f"GEMINI_API_KEY_{i}")
                if key:
                    keys.append(key)
            except Exception:
                pass
    except ImportError:
        pass

    # Also check environment variables
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)

    for i in range(1, 5):
        env_key_i = os.getenv(f"GEMINI_API_KEY_{i}")
        if env_key_i and env_key_i not in keys:
            keys.append(env_key_i)

    return keys


def _get_first_api_key() -> Optional[str]:
    """
    Return the first available Gemini API key (for backwards compatibility).
    """
    keys = _get_all_api_keys()
    return keys[0] if keys else None


def _clean_json_text(text: str) -> str:
    """
    Remove Markdown fences and surrounding whitespace from an LLM response.
    Keeps the content intact while making it easier to parse as JSON.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def invoke_and_parse_json(client, prompt: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Invoke an LLM client with `prompt`, strip any Markdown fences, and parse the
    response as a JSON object.

    Supports any client exposing either `generate_content(prompt)` (Gemini-style)
    or `invoke(prompt)` (LangChain-style). On any error, returns `default` when it
    is provided, otherwise re-raises. This is the single shared implementation of
    the "call the LLM, clean the response, parse JSON" boilerplate used by every
    *_llm.py agent.
    """
    try:
        if hasattr(client, "generate_content"):
            response = client.generate_content(prompt)
            if isinstance(response, str):
                response_text = response
            else:
                response_text = getattr(response, "text", None) or getattr(response, "content", str(response))
        elif hasattr(client, "invoke"):
            response = client.invoke(prompt)
            response_text = getattr(response, "content", None) or getattr(response, "text", str(response))
        else:
            raise AttributeError("Provided client doesn't match a recognized execution method.")

        return json.loads(_clean_json_text(str(response_text)))

    except Exception as e:
        logger.exception(f"Error parsing LLM output: {e}")
        if default is not None:
            return default
        raise


def _strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before closing braces/brackets to repair minor JSON issues.
    This is a safe, conservative cleanup to handle common LLM formatting mistakes.
    """
    previous = None
    current = text
    while current != previous:
        previous = current
        current = re.sub(r",\s*([}\]])", r"\1", current)
    return current


def build_gemini_client():
    """
    Build a configured Gemini client. Raises when API key or library is missing.
    """
    api_key = _get_first_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GEMINI_API_KEY_1..4 in .env file")
    if genai is None:
        raise ImportError("google-generativeai is not installed. Install with: pip install google-generativeai")

    genai.configure(api_key=api_key)
    logger.info(f"Gemini client initialized with model {model_name}")
    return GeminiClient(model=model_name)


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        self.api_keys = _get_all_api_keys()
        self.current_key_index = 0
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize client with current API key"""
        if self.current_key_index >= len(self.api_keys):
            raise ValueError("No valid API keys available for Gemini")

        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.client = genai.GenerativeModel(self.model)
        logger.info(f"Initialized Gemini client with API key {self.current_key_index + 1}/{len(self.api_keys)}")

    def _rotate_api_key(self):
        """Rotate to next available API key"""
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys exhausted")
        self._initialize_client()
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

    # Refactor - Helper Methods
    def _is_quota_error(self, error_str: str) -> bool:
        """Checks if the exception message matches common quota/rate-limit indicator"""
        quota_indicator = ["429", "quota", "rate limit", "exceeded", "per_minute", "per_day"]
        return any(indicator in error_str for indicator in quota_indicator)
    
    def _handle_quota_retry(self, attempt: int, max_retries: int, error_str: str) -> None:
        """Rotates keys for a retry attempts, or raises exhaustion error if out of retries"""
        logger.warn(f"Quota exceeded on key {self.current_key_index + 1}: {error_str}")
        
        if attempt >= max_retries:
            raise ValueError(f"All {len(self.api_keys)} API keys quota exceeded. Please wait before retrying.")
        
        try:
            self._rotate_api_key()
            logger.info(f"Retrying with next API key ({self.current_key_index + 1}/{len(self.api_keys)})")
        except ValueError as rotate_error:
            raise ValueError(f"All API keys exhausted or invalid: {rotate_error}")
    
    def _call_api_client(self, prompt: str, temperature: float, max_tokens: int, response_mime_type: Optional[str]) -> str:
        """Builds config, executes the client call and validates the response wrapper"""
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type

        response = self.client.generate_content(prompt, generation_config=generation_config)

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from LLM")

        return response.text

    # Refactor - Original Method
    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_mime_type: Optional[str] = None,
    ) -> str:
        """Generate content with automatic API key rotation on quota exceeded."""
        max_retries = len(self.api_keys)
        attempt = 0

        while attempt < max_retries:
            try:
                return self._call_api_client(prompt, temperature, max_tokens, response_mime_type)

            except Exception as e:
                error_str = str(e)

                if not self._is_quota_error(error_str):
                    logger.error(f"LLM API error: {error_str}")
                    raise ValueError(f"Failed to generate content from LLM: {error_str}")
                
                attempt += 1
                self._handle_quota_retry(attempt, max_retries, error_str)

        raise ValueError("Failed to generate content after all retries")

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        try:
            cleaned = _clean_json_text(response_text)
            candidates = [cleaned]

            # If the response contains other text, try extracting the first JSON object
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                candidates.append(json_match.group(0))

            last_error = None
            for candidate in candidates:
                try:
                    repaired = _strip_trailing_commas(candidate)
                    parsed_json = json.loads(repaired)
                    if not isinstance(parsed_json, dict):
                        raise ValueError("LLM response is not a valid JSON object")
                    return parsed_json
                except Exception as parse_error:
                    last_error = parse_error
                    continue

            raise ValueError(f"Invalid JSON in LLM response: {last_error}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
        except Exception as e:
            logger.error(f"Response extraction error: {str(e)}")
            raise ValueError(f"Failed to extract JSON from LLM response: {str(e)}")

    def validate_response_fields(self, response: Dict[str, Any], required_fields: list) -> None:
        missing_fields = [field for field in required_fields if field not in response]

        if missing_fields:
            raise ValueError(f"LLM response missing required fields: {missing_fields}")

    def generate_structured_json(
        self,
        prompt: str,
        required_fields: list,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON response with consistent validation and parsing safeguards.
        """
        response_text = self.generate_content(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_mime_type="application/json"
        )
        result = self.extract_json_from_response(response_text)
        self.validate_response_fields(result, required_fields)
        return result


def extract_json(text: str) -> Dict[str, Any]:
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
        return json.loads(json_match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")
