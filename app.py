"""
Entrypoint for Gradio, see https://gradio.app/
"""

import platform
import asyncio
import base64
import os
import io
import json
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, Dict
from PIL import Image

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors
from computer_use_demo.tools.logger import logger, truncate_string

logger.info("Starting the gradio app")

screens = get_monitors()
logger.info(f"Found {len(screens)} screens")

from computer_use_demo.loop import APIProvider, sampling_loop_sync

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.computer import get_screen_details
SCREEN_NAMES, SELECTED_SCREEN_INDEX = get_screen_details()

API_KEY_FILE = "./api_keys.json"

WARNING_TEXT = "⚠️ Security Alert: Do not provide access to sensitive accounts or data, as malicious web content can hijack Agent's behavior. Keep monitor on the Agent's actions."


def setup_state(state):

    if "messages" not in state:
        state["messages"] = []
    # -------------------------------
    if "planner_model" not in state:
        state["planner_model"] = "gpt-4o"  # default
    if "actor_model" not in state:
        state["actor_model"] = "ShowUI"    # default
    if "planner_provider" not in state:
        state["planner_provider"] = "openai"  # default
    if "actor_provider" not in state:
        state["actor_provider"] = "local"    # default

     # Fetch API keys from environment variables
    if "openai_api_key" not in state: 
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")    
    if "qwen_api_key" not in state:
        state["qwen_api_key"] = os.getenv("QWEN_API_KEY", "")
    if "openrouter_api_key" not in state:
        state["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY", "")
    if "lmstudio_url" not in state:
        state["lmstudio_url"] = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
    if "ui_tars_url" not in state:
        state["ui_tars_url"] = ""

    # Set the initial api_key based on the provider
    if "planner_api_key" not in state:
        if state["planner_provider"] == "openai":
            state["planner_api_key"] = state["openai_api_key"]
        elif state["planner_provider"] == "anthropic":
            state["planner_api_key"] = state["anthropic_api_key"]
        elif state["planner_provider"] == "qwen":
            state["planner_api_key"] = state["qwen_api_key"]
        elif state["planner_provider"] == "openrouter":
            state["planner_api_key"] = state["openrouter_api_key"]
        elif state["planner_provider"] == "lmstudio":
            state["planner_api_key"] = state["lmstudio_url"]
        else:
            state["planner_api_key"] = ""

    logger.info(f"loaded initial api_key for {state['planner_provider']}: {state['planner_api_key']}")

    if not state["planner_api_key"]:
        logger.warning("Planner API key not found. Please set it in the environment or paste in textbox.")


    if "selected_screen" not in state:
        state['selected_screen'] = SELECTED_SCREEN_INDEX if SCREEN_NAMES else 0

    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 10 # 10
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = ""
        # remove if want to use default system prompt
        device_os_name = "Windows" if platform.system() == "Windows" else "Mac" if platform.system() == "Darwin" else "Linux"
        state["custom_system_prompt"] += f"\n\nNOTE: you are operating a {device_os_name} machine"
    if "hide_images" not in state:
        state["hide_images"] = False
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
        
    if "showui_config" not in state:
        state["showui_config"] = "Default"
    if "max_pixels" not in state:
        state["max_pixels"] = 1344
    if "awq_4bit" not in state:
        state["awq_4bit"] = False


async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response


def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output


def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        logger.info(f"_render_message: {str(message)[:100]}")

        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
            or message.__class__.__name__ == "CLIResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return message.text
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            return f"Tool Use: {message.name}\nInput: {message.input}"
        else:  
            return message


    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))

    # Create a concise version of the chatbot state for logging
    concise_state = [(truncate_string(user_msg), truncate_string(bot_msg)) for user_msg, bot_msg in chatbot_state]
    logger.info(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")


def process_input(user_input, state):
    
    setup_state(state)

    # Append the user message to state["messages"]
    state["messages"].append(
            {
                "role": "user",
                "content": [TextBlock(type="text", text=user_input)],
            }
        )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        system_prompt_suffix=state["custom_system_prompt"],
        planner_model=state["planner_model"],
        planner_provider=state["planner_provider"],
        actor_model=state["actor_model"],
        actor_provider=state["actor_provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"]),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["planner_api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        selected_screen=state['selected_screen'],
        showui_max_pixels=state['max_pixels'],
        showui_awq_4bit=state['awq_4bit'],
        lmstudio_base_url=state["lmstudio_url"]
    ):  
        if loop_msg is None:
            yield state['chatbot_messages']
            logger.info("End of task. Close the loop.")
            break
            

        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    state = gr.State({})  # Use Gradio's state management
    setup_state(state.value)  # Initialize the state

    # Retrieve screen details
    gr.Markdown("# Computer Use OOTB")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(WARNING_TEXT)

    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                # --------------------------
                # Planner
                planner_model = gr.Dropdown(
                    label="Planner Model",
                    choices=["gpt-4o", 
                             "gpt-4o-mini", 
                             "qwen2-vl-max", 
                             "qwen2-vl-2b (local)", 
                             "qwen2-vl-7b (local)", 
                             "qwen2-vl-2b (ssh)", 
                             "qwen2-vl-7b (ssh)",
                             "qwen2.5-vl-7b (ssh)", 
                             "claude-3-5-sonnet-20241022",
                    value="gpt-4o",
                    interactive=True,
                )
            with gr.Column():
                planner_api_provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                )
            with gr.Column():
                planner_api_key = gr.Textbox(
                    label="Planner API Key",
                    type="password",
                    value=state.value.get("planner_api_key", ""),
                    placeholder="Paste your planner model API key",
                    interactive=True,
                )

            with gr.Column():
                actor_model = gr.Dropdown(
                    label="Actor Model",
                    choices=["ShowUI", "UI-TARS",
                             "LM Studio showui-2b",
                             "LM Studio ui-tars-7b-dpo",
                             "LM Studio ui-tars-2b-sft"],
                    value="ShowUI",
                    interactive=True,
                )

            with gr.Column():
                custom_prompt = gr.Textbox(
                    label="System Prompt Suffix",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                SCREEN_NAMES = screen_options
                SELECTED_SCREEN_INDEX = primary_index
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True,
                )
    
    with gr.Accordion("ShowUI Advanced Settings", open=False):  
        
        gr.Markdown("""
                    **Note:** Adjust these settings to fine-tune the resource (**memory** and **infer time**) and performance trade-offs of ShowUI. \\
                    Quantization model requires additional download. Please refer to [Computer Use OOTB - #ShowUI Advanced Settings guide](https://github.com/showlab/computer_use_ootb?tab=readme-ov-file#showui-advanced-settings) for preparation for this feature.
                    """)

        # New configuration for ShowUI
        with gr.Row():
            with gr.Column():
                showui_config = gr.Dropdown(
                    label="ShowUI Preset Configuration",
                    choices=["Default (Maximum)", "Medium", "Minimal", "Custom"],
                    value="Default (Maximum)",
                    interactive=True,
                )
            with gr.Column():
                max_pixels = gr.Slider(
                    label="Max Visual Tokens",
                    minimum=720,
                    maximum=1344,
                    step=16,
                    value=1344,
                    interactive=False,
                )
            with gr.Column():
                awq_4bit = gr.Checkbox(
                    label="Enable AWQ-4bit Model",
                    value=False,
                    interactive=False
                )
            
    # Define the merged dictionary with task mappings
    merged_dict = json.load(open("assets/examples/ootb_examples.json", "r"))

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
    
    # Callback to update the second dropdown based on the first selection
    def update_second_menu(selected_category):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).keys()))

    # Callback to update the third dropdown based on the second selection
    def update_third_menu(selected_category, selected_option):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys()))

    # Callback to update the textbox based on the third selection
    def update_textbox(selected_category, selected_option, selected_task):
        task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        return prompt, preview_image, task_hint
    
    # Function to update the global variable when the dropdown changes
    def update_selected_screen(selected_screen_name, state):
        global SCREEN_NAMES
        global SELECTED_SCREEN_INDEX
        SELECTED_SCREEN_INDEX = SCREEN_NAMES.index(selected_screen_name)
        logger.info(f"Selected screen updated to: {SELECTED_SCREEN_INDEX}")
        state['selected_screen'] = SELECTED_SCREEN_INDEX


    def update_planner_model(model_selection, state):
        state["model"] = model_selection
        # Update planner_model
        state["planner_model"] = model_selection
        logger.info(f"Model updated to: {state['planner_model']}")
        
        if model_selection == "qwen2-vl-max":
            provider_choices = ["qwen"]
            provider_value = "qwen"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "qwen API key"
            actor_model_choices = ["ShowUI", "UI-TARS"]
            actor_model_value = "ShowUI"
            actor_model_interactive = True
            api_key_type = "password"  # Display API key in password form
        
        elif model_selection in ["qwen2-vl-2b (local)", "qwen2-vl-7b (local)"]:
            # Set provider to "openai", make it unchangeable
            provider_choices = ["local"]
            provider_value = "local"
            provider_interactive = False
            api_key_interactive = False
            api_key_placeholder = "not required"
            actor_model_choices = ["ShowUI", "UI-TARS"]
            actor_model_value = "ShowUI"
            actor_model_interactive = True
            api_key_type = "password"  # Maintain consistency

        elif "ssh" in model_selection:
            provider_choices = ["ssh"]
            provider_value = "ssh"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "ssh host and port (e.g. localhost:8000)"
            actor_model_choices = ["ShowUI", "UI-TARS"]
            actor_model_value = "ShowUI"
            actor_model_interactive = True
            api_key_type = "text"  # Display SSH connection info in plain text
            # If SSH connection info already exists, keep it
            if "planner_api_key" in state and state["planner_api_key"]:
                state["api_key"] = state["planner_api_key"]
            else:
                state["api_key"] = ""

        elif model_selection == "gpt-4o" or model_selection == "gpt-4o-mini":
            # Set provider to "openai", make it unchangeable
            provider_choices = ["openai"]
            provider_value = "openai"
            provider_interactive = False
            api_key_interactive = True
            api_key_type = "password"  # Display API key in password form

            api_key_placeholder = "openai API key"
            actor_model_choices = ["ShowUI", "UI-TARS"]
            actor_model_value = "ShowUI"
            actor_model_interactive = True

        elif model_selection == "claude-3-5-sonnet-20241022":
            # Provider can be any of the current choices except 'openai'
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
            provider_value = "anthropic"  # Set default to 'anthropic'
            state['actor_provider'] = "anthropic"
            provider_interactive = True
            api_key_interactive = True
            api_key_placeholder = "claude API key"
            actor_model_choices = ["claude-3-5-sonnet-20241022"]
            actor_model_value = "claude-3-5-sonnet-20241022"
            actor_model_interactive = False
            api_key_type = "password"  # Display API key in password form

        elif model_selection == "OpenRouter qwen/qwen2.5-vl-72b-instruct:free":
            provider_choices = ["openrouter"]
            provider_value = "openrouter"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "OpenRouter API Key"
            actor_model_choices = ["ShowUI", "UI-TARS"] # Assuming compatible with standard actors
            actor_model_value = "ShowUI"
            actor_model_interactive = True
            api_key_type = "password"

        else:
            raise ValueError(f"Model {model_selection} not supported")

        # Update the provider in state
        state["planner_api_provider"] = provider_value
        
        # Update api_key in state based on the provider
        if provider_value == "openai":
            state["api_key"] = state.get("openai_api_key", "")
        elif provider_value == "anthropic":
            state["api_key"] = state.get("anthropic_api_key", "")
        elif provider_value == "qwen":
            state["api_key"] = state.get("qwen_api_key", "")
        elif provider_value == "local":
            state["api_key"] = ""
        elif provider_value == "openrouter":
            state["api_key"] = state.get("openrouter_api_key", "")
        elif provider_value == "lmstudio":
            state["api_key"] = state.get("lmstudio_url", "http://localhost:1234/v1")
        # SSH的情况已经在上面处理过了，这里不需要重复处理

        provider_update = gr.update(
            choices=provider_choices,
            value=provider_value,
            interactive=provider_interactive
        )

        # Update the API Key textbox
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"],
            interactive=api_key_interactive,
            type=api_key_type  # 添加 type 参数的更新
        )

        actor_model_update = gr.update(
            choices=actor_model_choices,
            value=actor_model_value,
            interactive=actor_model_interactive
        )

        logger.info(f"Updated state: model={state['planner_model']}, provider={state['planner_api_provider']}, api_key={state['api_key']}")
        return provider_update, api_key_update, actor_model_update
    
    def update_actor_model(actor_model_selection, state):
        state["actor_model"] = actor_model_selection
        logger.info(f"Actor model updated to: {state['actor_model']}")

    def update_api_key_placeholder(provider_value, model_selection):
        if model_selection == "claude-3-5-sonnet-20241022":
            if provider_value == "anthropic":
                return gr.update(placeholder="anthropic API key")
            elif provider_value == "bedrock":
                return gr.update(placeholder="bedrock API key")
            elif provider_value == "vertex":
                return gr.update(placeholder="vertex API key")
            else:
                return gr.update(placeholder="")
        elif model_selection == "gpt-4o + ShowUI":
            return gr.update(placeholder="openai API key")
        else:
            return gr.update(placeholder="")

    def update_system_prompt_suffix(system_prompt_suffix, state):
        state["custom_system_prompt"] = system_prompt_suffix
        
    # When showui_config changes, we set the max_pixels and awq_4bit accordingly.
    def handle_showui_config_change(showui_config_val, state):
        if showui_config_val == "Default (Maximum)":
            state["max_pixels"] = 1344
            state["awq_4bit"] = False
            return (
                gr.update(value=1344, interactive=False), 
                gr.update(value=False, interactive=False)
            )
        elif showui_config_val == "Medium":
            state["max_pixels"] = 1024
            state["awq_4bit"] = False
            return (
                gr.update(value=1024, interactive=False), 
                gr.update(value=False, interactive=False)
            )
        elif showui_config_val == "Minimal":
            state["max_pixels"] = 1024
            state["awq_4bit"] = True
            return (
                gr.update(value=1024, interactive=False), 
                gr.update(value=True, interactive=False)
            )
        elif showui_config_val == "Custom":
            # Do not overwrite the current user values, just make them interactive
            return (
                gr.update(interactive=True), 
                gr.update(interactive=True)
            )

    def update_api_key(api_key_value, state):
        """Handle API key updates"""
        state["planner_api_key"] = api_key_value
        if state["planner_provider"] == "ssh":
            state["api_key"] = api_key_value # This seems to be a bug, ssh uses planner_api_key directly.
        elif state["planner_provider"] == APIProvider.OPENROUTER:
            state["openrouter_api_key"] = api_key_value
            state["api_key"] = api_key_value
        elif state["planner_provider"] == APIProvider.LMSTUDIO:
            state["lmstudio_url"] = api_key_value
            state["api_key"] = api_key_value
        logger.info(f"API key updated: provider={state['planner_provider']}, api_key={state['planner_api_key']}") # Log planner_api_key

    with gr.Accordion("Quick Start Prompt", open=False):  # open=False 表示默认收
        # Initialize Gradio interface with the dropdowns
        with gr.Row():
            # Set initial values
            initial_category = "Game Play"
            initial_second_options = list(merged_dict[initial_category].keys())
            initial_third_options = list(merged_dict[initial_category][initial_second_options[0]].keys())
            initial_text_value = merged_dict[initial_category][initial_second_options[0]][initial_third_options[0]]

            with gr.Column(scale=2):
                # First dropdown for Task Category
                first_menu = gr.Dropdown(
                    choices=list(merged_dict.keys()), label="Task Category", interactive=True, value=initial_category
                )

                # Second dropdown for Software
                second_menu = gr.Dropdown(
                    choices=initial_second_options, label="Software", interactive=True, value=initial_second_options[0]
                )

                # Third dropdown for Task
                third_menu = gr.Dropdown(
                    choices=initial_third_options, label="Task", interactive=True, value=initial_third_options[0]
                    # choices=["Please select a task"]+initial_third_options, label="Task", interactive=True, value="Please select a task"
                )

            with gr.Column(scale=1):
                initial_image_value = "./assets/examples/init_states/honkai_star_rail_showui.png"  # default image path
                image_preview = gr.Image(value=initial_image_value, label="Reference Initial State", height=260-(318.75-280))
                hintbox = gr.Markdown("Task Hint: Selected options will appear here.")

        # Textbox for displaying the mapped value
        # textbox = gr.Textbox(value=initial_text_value, label="Action")

    # api_key.change(fn=lambda key: save_to_storage(API_KEY_FILE, key), inputs=api_key)

    with gr.Row():
        # submit_button = gr.Button("Submit")  # Add submit button
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")

    chatbot = gr.Chatbot(label="Chatbot History", type="tuples", autoscroll=True, height=580, group_consecutive_messages=False)
    
    planner_model.change(fn=update_planner_model, inputs=[planner_model, state], outputs=[planner_api_provider, planner_api_key, actor_model])
    planner_api_provider.change(fn=update_api_key_placeholder, inputs=[planner_api_provider, planner_model], outputs=planner_api_key)
    actor_model.change(fn=update_actor_model, inputs=[actor_model, state], outputs=None)

    screen_selector.change(fn=update_selected_screen, inputs=[screen_selector, state], outputs=None)
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    
    # When showui_config changes, we update max_pixels and awq_4bit automatically.
    showui_config.change(fn=handle_showui_config_change, 
                         inputs=[showui_config, state], 
                         outputs=[max_pixels, awq_4bit])
    
    # Link callbacks to update dropdowns based on selections
    first_menu.change(fn=update_second_menu, inputs=first_menu, outputs=second_menu)
    second_menu.change(fn=update_third_menu, inputs=[first_menu, second_menu], outputs=third_menu)
    third_menu.change(fn=update_textbox, inputs=[first_menu, second_menu, third_menu], outputs=[chat_input, image_preview, hintbox])

    # chat_input.submit(process_input, [chat_input, state], chatbot)
    submit_button.click(process_input, [chat_input, state], chatbot)

    planner_api_key.change(
        fn=update_api_key,
        inputs=[planner_api_key, state],
        outputs=None
    )

demo.launch(share=False,
            allowed_paths=["./"],
            server_port=7888)  # TODO: allowed_paths