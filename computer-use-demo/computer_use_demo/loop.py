"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""
import json
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""
<SYSTEM_CAPABILITY>
VERY IMPORTANT: WHEN SELECTING COORDINATES FOR THE MOUSE, MAKE SURE YOU ARE REFERENCING THE MOST RECENT SCREENSHOT ONLY

Note that your mouse pointer may have a yellow circle highlight it, so that humans can follow it.

* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You are an assistant who is using the Datadog web application. Do not take any actions outside of this application (which is used via firefox)
* If you do not see the datadog web application, you must ensure Firefox is open, then navigate to dd.datad0g.com to re-open it
* You may open new tabs as needed, but do not switch applications out of Firefox
* Firefox should already be open but if you need to open firefox, please just click on the firefox icon. Note, firefox-esr is what is installed on your system.
* If you're not sure what you're being asked to do, first take a screenshot, it might make more sense after that
* YOU MUST NOT respond saying "you need more information" unless you've already taken a screenshot to try gathering info
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
</SYSTEM_CAPABILITY>
"""

async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    do_print,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        # CrossRefTool(),
        # BashTool(),
        # EditTool(),
    )
    sys_prompt_text = f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    do_print("Debug: Using system prompt:")
    do_print(sys_prompt_text)
    system = BetaTextBlockParam(
        type="text",
        text=sys_prompt_text,
    )

    while True:
        enable_prompt_caching = False
        betas = [COMPUTER_USE_BETA_FLAG]
        image_truncation_threshold = 10
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            # _inject_prompt_caching(messages)
            # Is it ever worth it to bust the cache with prompt caching?
            image_truncation_threshold = 50
            system["cache_control"] = {"type": "ephemeral"}

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=tool_collection.to_params(),
                betas=betas,
            )
        except (APIStatusError, APIResponseValidationError) as e:
            api_response_callback(e.request, e.response, e)
            return messages
        except APIError as e:
            api_response_callback(e.request, e.body, e)
            return messages

        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        response = raw_response.parse()

        response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        inputs_this_loop = []
        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                tool_input = cast(dict[str, Any], content_block["input"])
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=tool_input,
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block["id"])
                )
                tool_output_callback(result, content_block["id"])
                inputs_this_loop.append(tool_input)

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})

        do_print("Loop Tool Inputs")
        do_print(inputs_this_loop)
        for in_args in inputs_this_loop:
            action = in_args.get("action")
            if action == "key" or action == "type" or action == "left_click":
                do_print("DO CROSS REF")
                messages = await do_cross_reference(client, model, messages, api_response_callback, output_callback, tool_collection, betas)
                break

# Other try
# ESSENTIAL_REASONING_STEPS
# Always double check EACH STEP that you perform, to ensure that the changes you made have the intended effect, and that the UI is reflecting your changes.
# Always return a brief chain of thought with your message.
# For EACH "click" action AND EACH "key" action AND EACH "type" action that you request, YOU MUST include a message describing the change you expect to see in the UI.
# When you're sent a screenshot after taking an action, YOU MUST check the screenshot against your previous expectations, to see if you need to try a different approach.


cross_reference_system_prompt = """
You have requested some number of "computer use" actions, which include mouse movements, mouse clicks, typing, and keystrokes.
You have requested these in order to accomplish some goal in the computer's UI. 
Your job is now to return 2 statements:
* What was the intended change in the UI as a result of the most recent actions?
* What actually happened according to the most recent screenshot - is that change reflected, or did something go wrong?

Please return in the following format:

Intended Change: <your explanation here>
Actual Result: <your detailed description of the UI>

Example 1:
Intended Change: I attempted to click on the button to edit a widget, which I expect would open some new edit view in the UI
Actual Result: The UI now shows a modal with an "Edit Widget" title, several fields that look modifiable, and a "Save" button, so it seems that this was successful

Example 2:
Intended Change: I tried to enter the text "CPU Usage by Pod" into the title field, so I expect that text to now be visible inside the title field
Actual Result: This text is not visible in the title field, which looks the same as before, so this action was not successful

DO NOT repeat what actions you've already tried. YOU MUST NOT repeat "Requested this action..." messages. Just reflect on what happened in the UI.
"""

async def do_cross_reference(client, model, messages, api_response_callback, output_callback, tool_collection, betas):
    system = BetaTextBlockParam(
        type="text",
        text=cross_reference_system_prompt,
    )
    transformed_messages = [remove_tool(msg) for msg in messages]
    try:
        raw_response = client.beta.messages.with_raw_response.create(
            max_tokens=4096,
            messages=transformed_messages,
            model=model,
            system=[system],
            # tools=tool_collection.to_params(),
            # betas=betas,
        )
    except (APIStatusError, APIResponseValidationError) as e:
        api_response_callback(e.request, e.response, e)
        return messages
    except APIError as e:
        api_response_callback(e.request, e.body, e)
        return messages
    api_response_callback(
        raw_response.http_response.request, raw_response.http_response, None
    )

    response = raw_response.parse()

    response_params = _response_to_params(response)
    for content_block in response_params:
        output_callback(content_block)
    messages.append(
        {
            "role": "assistant",
            "content": response_params,
        }
    )
    sc = await ComputerTool()(action="screenshot")
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Use this analysis along with this updated screenshot to decide on next actions"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": sc.base64_image,
                    },
                }
            ],
        }
    )
    return messages

def remove_tool(message: BetaMessageParam) -> BetaMessageParam:
    if isinstance(message["content"], str):
        return message
    elif isinstance(message["content"], list):
        new = {"role": message["role"]}
        new["content"] = [tool_result_to_standard_content(content) for content in message["content"]]
        return new
    else:
        return message

def tool_result_to_standard_content(content):
    if isinstance(content, dict) and content["type"] == "tool_result":
        return content["content"][0]
    elif isinstance(content, dict) and content["type"] == "tool_use":
        return {
            "type": "text",
            "text": f"""Requested this action to accomplish my goal: {json.dumps(content["input"])}"""
        }
    else:
        return content

def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
