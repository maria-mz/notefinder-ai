import base64
import json
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from notefinder.prompt import Prompt


class BadOutputError(Exception):
    """Claude replied, but not with usable output."""


@dataclass
class Extraction:
    transcript: str
    concepts: list[str]
    summary: str
    input_tokens: int
    output_tokens: int


# Structured outputs guarantee a reply that finishes normally matches this
# schema, so parse_output only checks what the schema can't express.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["transcript", "concepts", "summary"],
    "additionalProperties": False,
}


def make_client(max_retries: int) -> AsyncAnthropic:
    return AsyncAnthropic(max_retries=max_retries)


def parse_output(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BadOutputError(f"invalid JSON: {e}") from e

    if not data["transcript"].strip():
        raise BadOutputError("'transcript' must be non-empty")

    if not (data["concepts"] and all(c.strip() for c in data["concepts"])):
        raise BadOutputError("'concepts' must be a non-empty list of non-empty strings")

    if not data["summary"].strip():
        raise BadOutputError("'summary' must be non-empty")

    return data


class Extractor:
    def __init__(self, client: AsyncAnthropic, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    async def extract(self, png_bytes: bytes, prompt: Prompt) -> Extraction:
        image_b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt.text},
                ],
            }],
        )
        if response.stop_reason != "end_turn":
            raise BadOutputError(f"stopped early: {response.stop_reason}")

        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise BadOutputError("response contained no text block")

        data = parse_output(text)
        return Extraction(
            transcript=data["transcript"],
            concepts=data["concepts"],
            summary=data["summary"],
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
