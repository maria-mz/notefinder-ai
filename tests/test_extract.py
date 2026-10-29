import asyncio
import json
import base64
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from extract import BadOutputError, extract_page, parse_output

# Constants

GOOD = {
    "transcript": "∀x P(x)",
    "concepts": ["Predicate logic", "Universal quantifier"],
    "summary": "Predicate logic notes.",
}

PNG = b"\x89PNG fake image bytes"
PROMPT = "Transcribe this page."


# Helpers

def text_block(text):
    return NS(type="text", text=text)


def thinking_block():
    return NS(type="thinking")


def make_response(content, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    return NS(
        content=content,
        stop_reason=stop_reason,
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def make_client(content, **kwargs):
    response = make_response(content, **kwargs)
    return NS(messages=NS(create=AsyncMock(return_value=response)))


def call(client, **kwargs):
    return asyncio.run(extract_page(client, PNG, PROMPT, **kwargs))


def without(key):
    return {k: v for k, v in GOOD.items() if k != key}


# parse_output

def test_parse_output_accepts_valid_json():
    assert parse_output(json.dumps(GOOD)) == GOOD


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "```\n{}\n```"])
def test_parse_output_strips_code_fences(wrapper):
    assert parse_output(wrapper.format(json.dumps(GOOD))) == GOOD


def test_parse_output_rejects_invalid_json():
    with pytest.raises(BadOutputError, match="invalid JSON"):
        parse_output("not json at all")


def test_parse_output_rejects_non_object():
    with pytest.raises(BadOutputError, match="not a JSON object"):
        parse_output("[1, 2, 3]")


@pytest.mark.parametrize(
    "data, field",
    [
        (without("transcript"), "transcript"),
        ({**GOOD, "transcript": 5}, "transcript"),
        (without("concepts"), "concepts"),
        ({**GOOD, "concepts": []}, "concepts"),
        ({**GOOD, "concepts": ["ok", 3]}, "concepts"),
        ({**GOOD, "concepts": ["ok", "  "]}, "concepts"),
        (without("summary"), "summary"),
        ({**GOOD, "summary": "   "}, "summary"),
        ({**GOOD, "transcript": ""}, "transcript"),
        ({**GOOD, "transcript": "  \n"}, "transcript"),
    ]
)
def test_parse_output_rejects_bad_fields(data, field):
    with pytest.raises(BadOutputError, match=field):
        parse_output(json.dumps(data))


# extract_page

def test_extract_page_returns_fields_and_token_counts():
    client = make_client(
        [text_block(json.dumps(GOOD))],
        input_tokens=3519,
        output_tokens=686,
    )

    result = call(client)

    assert result.transcript == GOOD["transcript"]
    assert result.concepts == GOOD["concepts"]
    assert result.summary == GOOD["summary"]
    assert result.input_tokens == 3519
    assert result.output_tokens == 686


def test_extract_page_skips_thinking_blocks():
    client = make_client([
        thinking_block(),
        text_block(json.dumps(GOOD)),
    ])

    assert call(client).summary == GOOD["summary"]


def test_extract_page_rejects_truncated_output():
    client = make_client(
        [text_block('{"transcript": "cut off')],
        stop_reason="max_tokens",
    )

    with pytest.raises(BadOutputError, match="max_tokens"):
        call(client)


def test_extract_page_rejects_response_without_text():
    client = make_client([thinking_block()])

    with pytest.raises(BadOutputError, match="no text block"):
        call(client)


def test_extract_page_propagates_bad_json():
    client = make_client([text_block("Sorry, I can't read this page.")])

    with pytest.raises(BadOutputError, match="invalid JSON"):
        call(client)


def test_extract_page_sends_expected_message():
    client = make_client([text_block(json.dumps(GOOD))])
    call(client, model="some-model")

    sent = client.messages.create.call_args.kwargs

    assert sent["model"] == "some-model"
    assert sent["messages"] == [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(PNG).decode("ascii"),
                },
            },
            {
                "type": "text",
                "text": PROMPT,
            },
        ],
    }]
