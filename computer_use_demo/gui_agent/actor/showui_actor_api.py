import os
import base64
from openai import OpenAI

from computer_use_demo.gui_agent.llm_utils.oai import encode_image
from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.tools.logger import logger, truncate_string
from computer_use_demo.tools.colorful_text import colorful_text_showui

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ShowUIActorAPI:
    _NAV_SYSTEM = """
    You are an assistant trained to navigate the {_APP} screen.
    Given a task instruction, a screen observation, and an action history sequence,
    output the next action and wait for the next observation.
    Here is the action space:
    {_ACTION_SPACE}
    """

    _NAV_FORMAT = """
    Format the action as a dictionary with the following keys:
    {'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

    If value or position is not applicable, set it as None.
    Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
    Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
    """

    action_map = {
    'desktop': """
        1. CLICK: Click on an element, value is not applicable and the position [x,y] is required.
        2. INPUT: Type a string into an element, value is a string to type and the position [x,y] is required.
        3. HOVER: Hover on an element, value is not applicable and the position [x,y] is required.
        4. ENTER: Enter operation, value and position are not applicable.
        5. SCROLL: Scroll the screen, value is the direction to scroll and the position is not applicable.
        6. ESC: ESCAPE operation, value and position are not applicable.
        7. PRESS: Long click on an element, value is not applicable and the position [x,y] is required.
        """
    # 'phone' action space could be added if needed
    }

    def __init__(self, base_url: str, model_name: str, output_callback, api_key: str = "", selected_screen: int = 0, split: str = 'desktop'):
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.selected_screen = selected_screen
        self.output_callback = output_callback
        self.split = split # 'desktop' or 'phone'

        self.system_prompt = self._NAV_SYSTEM.format(
            _APP=self.split,
            _ACTION_SPACE=self.action_map[self.split]
        )
        self.action_history = '' # Initialize action history

    def __call__(self, messages):
        task = messages # In planner+actor mode, messages from planner is the task for actor

        # Get screenshot
        screenshot_pil, screenshot_path_obj = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=1920, target_height=1080)
        screenshot_path = str(screenshot_path_obj)
        image_base64 = encode_image(screenshot_path)

        if self.output_callback:
            self.output_callback(f'Screenshot for API-based {colorful_text_showui} ({self.model_name}):\n<img src="data:image/png;base64,{image_base64}">', sender="bot")

        # Construct messages for the API
        # Similar to original ShowUIActor, considering action history
        # The prompt structure might need adjustment based on how the API-served model is fine-tuned.
        # Assuming a general instruction, task, and optional history.

        user_content = []
        # System prompt is handled by the API call structure for OpenAI compatible APIs
        # user_content.append({"type": "text", "text": self.system_prompt + self._NAV_FORMAT}) # System prompt included here for now
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

        current_prompt = f"Task: {task}"
        if self.action_history:
            current_prompt += f"\n\nPrevious Actions:\n{self.action_history}"
        current_prompt += f"\n\nGiven the screenshot and the task, provide the next action based on the defined action space and format."

        user_content.append({"type": "text", "text": current_prompt})

        logger.info(f"Sending messages to ShowUI API model {self.model_name} on {self.base_url}: Task: {task}")

        api_messages = [
            {"role": "system", "content": self.system_prompt + "\n" + self._NAV_FORMAT},
            {"role": "user", "content": user_content}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            max_tokens=128, # Max tokens for action generation
            temperature=0 # Deterministic output for actions
        )

        output_text = response.choices[0].message.content

        # Update action history
        # Assuming the model directly outputs the action string like "{'action': 'CLICK', ...}"
        # If it includes "Action: ", that needs to be stripped.
        # For now, assume direct output of the dictionary-like string.
        self.action_history += output_text + '\n'

        logger.info(f"Received action from {self.model_name}: {truncate_string(output_text)}")

        # Return response in the format ShowUIExecutor expects
        # The original ShowUIActor returns: {'content': output_text, 'role': 'assistant'}
        # where output_text is a string like "{'action': 'CLICK', ...}"
        return {'content': output_text, 'role': 'assistant'}
