import base64
import json
from dataclasses import dataclass

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8192


class BadOutputError(Exception):
    """Claude replied, but not with usable output."""


@dataclass
class Extraction:
    transcript: str
    concepts: list[str]
    summary: str
    input_tokens: int
    output_tokens: int


def make_client() -> AsyncAnthropic:
    return AsyncAnthropic(max_retries=5)


def parse_output(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):  # models sometimes wrap JSON in a code fence
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BadOutputError(f"invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise BadOutputError("not a JSON object")

    transcript = data.get("transcript")
    concepts = data.get("concepts")
    summary = data.get("summary")

    if not (isinstance(transcript, str) and transcript.strip()):
        raise BadOutputError("'transcript' must be a non-empty string")

    if not (isinstance(concepts, list) and concepts
            and all(isinstance(c, str) and c.strip() for c in concepts)):
        raise BadOutputError("'concepts' must be a non-empty list of strings")

    if not (isinstance(summary, str) and summary.strip()):
        raise BadOutputError("'summary' must be a non-empty string")

    return data


async def extract_page(client: AsyncAnthropic, png_bytes: bytes, prompt: str,
                       model: str = MODEL) -> Extraction:
    image_b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    response = await client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                },
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
