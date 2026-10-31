"""Reading a page image with Claude: its transcript, concepts, and summary."""
import base64
import json
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from notefinder.prompt import Prompt


class BadOutputError(Exception):
    """Claude replied, but not with usable output."""


@dataclass
class Extraction:
    """What Claude read from one page.

    :ivar transcript: The page's text, transcribed.
    :ivar concepts: The topics the page covers.
    :ivar summary: A short summary of the page.
    :ivar input_tokens: Tokens Claude read, for cost tracking.
    :ivar output_tokens: Tokens Claude wrote, for cost tracking.
    """
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
    """Create an Anthropic client. The API key comes from ``ANTHROPIC_API_KEY``.

    :param max_retries: How many times the client retries a failed request.
    :returns: The client.
    """
    return AsyncAnthropic(max_retries=max_retries)


def parse_output(text: str) -> dict:
    """Parse Claude's reply and check what :data:`OUTPUT_SCHEMA` can't.

    :param text: The reply's text, expected to match :data:`OUTPUT_SCHEMA`.
    :returns: The parsed reply.
    :raises BadOutputError: If ``text`` isn't JSON, or a field is empty.
    """
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
    """Reads page images with Claude.

    :ivar client: The Anthropic client.
    :ivar model: The Claude model to use.
    :ivar max_tokens: The most tokens Claude may write per page.
    """
    def __init__(self, client: AsyncAnthropic, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    async def extract(self, png_bytes: bytes, prompt: Prompt) -> Extraction:
        """Read one page image.

        :param png_bytes: The page, rendered to PNG.
        :param prompt: The instructions to send with the image.
        :returns: What Claude read.
        :raises BadOutputError: If Claude stops early or its reply is unusable.
        :raises anthropic.APIError: If the request fails after the client's retries.
        """
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
