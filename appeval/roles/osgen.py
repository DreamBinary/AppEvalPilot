#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName : osgen.py
# @Time : 2025/05/28 14:42
# @Author : fiv

import asyncio
import copy
import json
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleContext
from metagpt.schema import AIMessage, Message
from metagpt.utils.common import encode_image
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.actions.reflection import Reflection
from appeval.actions.screen_info_extractor import ScreenInfoExtractor
from appeval.prompts.osagent import ActionPromptContext, Android_prompt, PC_prompt
from appeval.tools.chrome_debugger import ChromeDebugger
from appeval.tools.device_controller import ControllerTool
from appeval.tools.icon_detect import IconDetectTool
from appeval.tools.ocr import OCRTool
from tqdm import tqdm
from pywinauto import Desktop
        

# 忽略所有警告
warnings.filterwarnings("ignore")


class OSAgentContext(RoleContext):
    """Runtime context for OSAgent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # thought: str = ""  # Current thinking content
    # thought_history: List[str] = Field(default_factory=list)  # Historical thinking records list
    # summary_history: List[str] = Field(default_factory=list)  # Historical operation summary list
    action_history: List[str] = Field(default_factory=list)  # Historical executed action list
    # reflection_thought_history: List[str] = Field(default_factory=list)  # Historical reflection records list
    # reflection_thought: str = ""  # Current reflection content
    # summary: str = ""  # Current operation summary
    action: str = ""  # Current executed action
    # task_list: str = ""  # Task list
    # completed_requirements: str = ""  # Completed requirements
    # memory: List[str] = Field(default_factory=list)  # Important content memory list
    # error_flag: bool = False  # Error flag
    # iter: int = 0  # Current iteration count
    perception_infos: List[Dict] = Field(default_factory=list)  # Current perception information list
    last_perception_infos: List[Dict] = Field(default_factory=list)  # Previous perception information list
    # width: int = 0  # Screen width
    # height: int = 0  # Screen height
    webbrowser_console_logs: List[Any] = Field(default_factory=list)  # Browser console log list
    screenshot_origin_file: str = ""  # Last original screenshot file path
    screenshot_draw_file: str = ""  # Last screenshot with visualization boxes file path
    init_action_list: List[str] = Field(default_factory=list)  # Initial action list
    classified_perception_infos: List[Dict] = Field(default_factory=list)  # Classified perception infos
    next_perception_infos: List[Dict] = Field(default_factory=list)  # Next perception infos
    completion_status: bool = False  # Completion status
    curr_time: str = ""  # Current time

    def reset(self) -> None:
        """Reset all states to initial values"""
        # self.thought = ""
        # self.thought_history = []
        # self.summary_history = []
        self.action_history = []
        # self.reflection_thought_history = []
        # self.reflection_thought = ""
        # self.summary = ""
        self.action = ""
        # self.task_list = ""
        # self.completed_requirements = ""
        # self.memory = []
        # self.error_flag = False
        # self.iter = 0
        self.perception_infos = []
        self.last_perception_infos = []
        # self.width = 0
        # self.height = 0
        self.webbrowser_console_logs = []
        self.screenshot_origin_file = ""
        self.screenshot_draw_file = ""
        self.init_action_list = [
            "Run (pyautogui.press('win'); time.sleep(1));",
            "Run (pyperclip.copy(\"\"\"firefox\"\"\"); time.sleep(0.5); pyautogui.hotkey('ctrl', 'v'); time.sleep(1); pyautogui.press('enter'); time.sleep(2))",
        ]
        self.classified_perception_infos = []
        self.next_perception_infos = []
        self.completion_status = True

    def save_state(self, path: Optional[Path | str]) -> None:
        # all attributes
        state = {
            "completion_status": self.completion_status,
            "init_action_list": self.init_action_list,
            "action_history": self.action_history,
            "next_perception_infos": self.next_perception_infos,
            "classified_perception_infos": self.classified_perception_infos,
            "perception_infos": self.perception_infos,
            "action": self.action,
            "screenshot_origin_file": self.screenshot_origin_file,
            "screenshot_draw_file": self.screenshot_draw_file,
            "last_perception_infos": self.last_perception_infos,
            "webbrowser_console_logs": self.webbrowser_console_logs,
            "curr_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }

        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
            
    def load_state(self, path: Optional[Path | str]) -> None:
        """Load state from a file"""
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file {path} does not exist.")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        for key, value in state.items():
            setattr(self, key, value)
        logger.info(f"State loaded from {path}")

    # def dump_iter(self, path: Optional[Path | str] = None, **kwargs) -> Dict[str, Any]:
    #     """Dump current iteration context to a dictionary"""
    #     result =  {
    #         "iter": self.iter,
    #         "action": self.action,
    #         "thought": self.thought,
    #         "summary": self.summary,
    #         "task_list": self.task_list,
    #         "perception_infos": copy.deepcopy(self.perception_infos),
    #         "last_perception_infos": copy.deepcopy(self.last_perception_infos),
    #         "reflection_thought": self.reflection_thought,
    #         "init_action_list": self.init_action_list,
    #         "image_info": {
    #             "width": self.width,
    #             "height": self.height,
    #             "origin_file": self.screenshot_origh_file,
    #             "draw_file": self.screenshot_draw_file,
    #         },
    #         **kwargs,
    #     }
    #     if path:
    #         if isinstance(path, str):
    #             path = Path(path)
    #         path.parent.mkdir(parents=True, exist_ok=True)
    #         with open(path, "w", encoding="utf-8") as f:
    #             json.dump(result, f, ensure_ascii=False, indent=4)
    #     return result


class OSAgent(Role):
    """Operating System Agent class for executing automated tasks"""

    name: str = "OSAgent"
    profile: str = "OS Agent"
    goal: str = "Execute automated tasks"
    constraints: str = "Ensure task execution accuracy and efficiency"
    desc: str = "Operating System Agent class for executing automated tasks"

    rc: OSAgentContext = Field(default_factory=OSAgentContext)

    def __init__(
        self,
        # Basic configuration parameters
        platform: str = "Android",
        max_iters: int = 50,
        # Feature switch parameters
        use_ocr: bool = True,
        quad_split_ocr: bool = False,
        use_icon_detect: bool = True,
        use_icon_caption: bool = True,
        use_memory: bool = True,
        use_reflection: bool = True,
        use_som: bool = False,
        extend_xml_infos: bool = True,
        use_chrome_debugger: bool = False,
        # Display and layout parameters
        location_info: str = "center",
        draw_text_box: bool = False,
        # Path related parameters
        log_dirs: str = "workspace",
        font_path: str = str(Path(__file__).parent / "simhei.ttf"),
        knowledge_base_path: str = str(Path(__file__).parent),
        # Other optional parameters
        system_prompt: str = "",
        add_info: str = "",
        **kwargs,
    ) -> None:
        """Initialize OSAgent.

        Args:
            platform (str): Operating system type (Windows, Mac, or Android).
            max_iters (int): Maximum number of iterations.
            use_ocr (bool): Whether to use OCR.
            quad_split_ocr (bool): Whether to split image into four parts for OCR recognition.
            use_icon_detect (bool): Whether to use icon detection.
            use_icon_caption (bool): Whether to use icon caption.
            use_memory (bool): Whether to enable important content memory.
            use_reflection (bool): Whether to perform reflection.
            use_som (bool): Whether to draw visualization boxes on screenshots.
            extend_xml_infos (bool): Whether to add XML element information.
            use_chrome_debugger (bool): Whether to record browser console output.
            location_info (str): Location information type (center or bbox).
            draw_text_box (bool): Whether to draw text boxes in visualization.
            log_dirs (str): Log directory
            font_path (str): Font path.
            knowledge_base_path (str): Preset knowledge base file directory path
            system_prompt (str): System prompt
            add_info (str): Additional information to add to the prompt
        """
        super().__init__(**kwargs)

        # Save configuration parameters
        self._init_config(locals())

        # Initialize environment
        self._init_environment()

        # Initialize tools
        self._init_tools()

    def _init_config(self, params: dict) -> None:
        """Initialize configuration parameters"""
        # Filter out self and kwargs
        config_params = {k: v for k, v in params.items() if k not in ["self", "kwargs"]}
        for key, value in config_params.items():
            setattr(self, key, value)

        # Set default additional prompt information
        if not self.add_info:
            self.add_info = self._get_default_add_info()

    def _get_default_add_info(self) -> str:
        """Get default additional prompt information"""
        if self.platform == "Windows":
            return (
                "If you need to interact with elements outside of a web popup, such as calendar or time selection "
                "popups, make sure to close the popup first. If the content in a text box is entered incorrectly, "
                "use the select all and delete actions to clear it, then re-enter the correct information. "
                "To open a folder in File Explorer, please use a double-click. "
            )
        elif self.platform == "Android":
            return (
                "If you need to open an app, prioritize using the Open app (app name) action. If this fails, "
                "return to the home screen and click the app icon on the desktop. If you want to exit an app, "
                "return to the home screen. If there is a popup ad in the app, you should close the ad first. "
                "If you need to switch to another app, you should first return to the desktop. When summarizing "
                "content, comparing items, or performing cross-app actions, remember to leverage the content in memory. "
            )
        return ""

    def _init_environment(self) -> None:
        """Initialize runtime environment"""
        # Initialize paths
        self._get_timestamped_paths()

        # Initialize logs
        self._setup_logs()

        # Initialize operating system environment
        self._init_os_env()

    def _init_tools(self) -> None:
        """Initialize tool components"""
        # Initialize information extractor
        self.info_extractor = ScreenInfoExtractor(platform=self.platform)

        # Initialize reflection tool
        self.reflection_action = Reflection(platform=self.platform)

        # Initialize icon detection/caption tool
        if self.use_icon_detect or self.use_icon_caption:
            self.icon_tool = IconDetectTool(self.llm)

        # Initialize OCR tool
        if self.use_ocr:
            self.ocr_tool = OCRTool()

        # Initialize browser debugger
        if self.use_chrome_debugger:
            self.chrome_debugger = ChromeDebugger()

    def _get_timestamped_paths(self) -> None:
        """Update file paths with timestamps"""
        current_time = time.strftime("%Y%m%d%H%M")

        # Base paths
        log_dir = Path(self.log_dirs) / current_time
        self.save_info = str(log_dir / "info.txt")
        self.save_img = log_dir / "images"
        self.save_img.mkdir(parents=True, exist_ok=True)

        # Screenshot related paths
        self.screenshot_dir = log_dir / "screenshot"
        self.screenshot_file = str(self.screenshot_dir / "screenshot.jpg")
        self.screenshot_som_file = str(self.screenshot_dir / "screenshot_som.png")
        self.last_screenshot_file = str(self.screenshot_dir / "last_screenshot.jpg")
        self.last_screenshot_som_file = str(self.screenshot_dir / "last_screenshot_som.png")
        
        # Trace dir
        self.trace_dir = log_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_dir = log_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _init_os_env(self) -> None:
        """Initialize operating system environment.

        Initialize corresponding controller and prompt tools based on different platforms (Android/Windows/Mac).
        """
        platform_configs = {
            "Android": {"controller_args": {"platform": "Android"}, "prompt_class": Android_prompt},
            "Windows": {
                "controller_args": {
                    "platform": "Windows",
                    "search_keys": ["win", "s"],
                    "ctrl_key": "ctrl",
                    "pc_type": "Windows",
                },
                "prompt_class": PC_prompt,
            },
            "Mac": {
                "controller_args": {
                    "platform": "Mac",
                    "search_keys": ["command", "space"],
                    "ctrl_key": "command",
                    "pc_type": "Mac",
                },
                "prompt_class": PC_prompt,
            },
        }

        if self.platform not in platform_configs:
            raise ValueError(f"Unsupported platform: {self.platform}")

        config = platform_configs[self.platform]
        logger.info(f"Initializing controller: {config['controller_args']}")
        self.controller = ControllerTool(**config["controller_args"])
        self.prompt_utils = config["prompt_class"]()

    def _reset_state(self) -> None:
        """Reset state, clear previous records when running new tasks"""
        # Reset state in rc
        self.rc.reset()

        # Reset temporary files and directories
        self._get_timestamped_paths()

        # Reset other states
        self.run_action_failed = False
        self.run_action_failed_exception = ""

        if self.use_chrome_debugger:
            self.chrome_debugger.start_monitoring()

        # Recreate screenshot directory
        if self.screenshot_dir.exists():
            shutil.rmtree(self.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logs(self) -> None:
        """Set up logging"""
        log_dir = Path(self.save_info).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Remove previously existing log handlers
        logger.remove()

        # Define log format
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | " "{level:<8} | " "{module}:{function}:{line} - " "{message}"

        # Add file log handler
        logger.add(
            self.save_info,
            level="DEBUG",
            format=log_format,
            mode="w",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # Add console log handler
        logger.add(sys.stdout, level="DEBUG", format=log_format, colorize=True, enqueue=True)

        logger.info(f"Initialized logging, log file: {self.save_info}")

    def _draw_bounding_boxes(
        self, image_path: str, coordinates: List[List[int]], output_path: str, font_path: str
    ) -> None:
        """Draw numbered coordinate boxes on the image.

        Args:
            image_path (str): Image path.
            coordinates (list): List of coordinate boxes, each box is a list of four elements [x1, y1, x2, y2].
            output_path (str): Output image path.
            font_path (str): Font path.
        """
        # Open image and get dimensions
        image = Image.open(image_path)
        height = image.size[1]

        # Calculate drawing parameters
        line_width = int(height * 0.0025)
        font_size = int(height * 0.012)
        text_offset_x = line_width
        text_offset_y = int(height * 0.013)

        # Generate random colors for each bounding box
        colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(coordinates))]

        # Draw bounding boxes and numbers
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        for i, (coord, color) in enumerate(zip(coordinates, colors)):
            # Draw bounding box using RGB color directly
            draw.rectangle(coord, outline=color, width=line_width)

            # Calculate text position and draw number
            text_x = coord[0] + text_offset_x
            text_y = max(0, coord[1] - text_offset_y)
            draw.text((text_x, text_y), str(i + 1), fill=color, font=font)

        # Save result
        image.convert("RGB").save(output_path)

    def _save_iteration_images(self, iter_num: int) -> None:
        """Save original and annotated images for current iteration.

        Args:
            iter_num: Current iteration number
        """
        # Build file paths
        origin_path = str(Path(self.save_img) / f"origin_{iter_num}.jpg")
        draw_path = str(Path(self.save_img) / f"draw_{iter_num}.jpg")

        # Copy image files
        shutil.copy2(self.screenshot_file, origin_path)
        shutil.copy2(self.output_image_path, draw_path)
        self.rc.screenshot_origin_file = origin_path
        self.rc.screenshot_draw_file = draw_path

    def _update_screenshot_files(self) -> None:
        """Update screenshot files"""
        # Update normal screenshot
        last_screenshot = Path(self.last_screenshot_file)
        if last_screenshot.exists():
            last_screenshot.unlink()
        Path(self.screenshot_file).rename(last_screenshot)

        # Update SOM screenshot
        if self.use_som:
            last_screenshot_som = Path(self.last_screenshot_som_file)
            if last_screenshot_som.exists():
                last_screenshot_som.unlink()
            Path(self.screenshot_som_file).rename(last_screenshot_som)

    def _check_last_three_start_with_wait(self, string_list: List[str]) -> bool:
        """Check if the last three strings in the list start with "Wait".

        Args:
            string_list (list): List of strings.

        Returns:
            bool: Returns True if the last three strings start with "Wait", False otherwise.
        """
        if len(string_list) < 3:
            return False
        return all(s.startswith("Wait") for s in string_list[-3:])

    def _get_app_info(self) -> Optional[str]:
        """Get application auxiliary information from preset app_info.json file."""
        info_path = Path(self.knowledge_base_path) / "app_info.json"
        if not info_path.exists():
            return None
        app_info = json.loads(info_path.read_text(encoding="utf-8"))
        package_name = self.controller.get_current_app_package()
        if not package_name:
            return None
        return app_info.get(package_name, None)

    async def _get_perception_infos(
        self, screenshot_file: str, screenshot_som_file: str
    ) -> Tuple[List[Dict[str, Any]], int, int, str]:
        """Get perception information, including OCR and icon detection.
        Args:
            screenshot_file (str): Screenshot file path.
            screenshot_som_file (str): Screenshot file path with visualization boxes.
        Returns:
            tuple: Tuple containing perception information list, image width, image height and output image path.
        """
        # Get screen screenshot
        self.controller.get_screenshot(screenshot_file)
        # Get screen screenshot width and height
        width, height = Image.open(screenshot_file).size

        # OCR processing
        text, text_coordinates = [], []
        if self.use_ocr:
            text, text_coordinates = self.ocr_tool.ocr(screenshot_file, split=self.quad_split_ocr)

        # Icon detection
        icon_coordinates = []
        if self.use_icon_detect:
            icon_coordinates = self.icon_tool.detect(screenshot_file)

        # Process output image
        output_image_path = screenshot_som_file
        if self.use_ocr and self.use_icon_detect and self.draw_text_box:
            rec_list = text_coordinates + icon_coordinates
            self._draw_bounding_boxes(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, self.font_path)
        elif self.use_icon_detect:
            self._draw_bounding_boxes(
                screenshot_file, copy.deepcopy(icon_coordinates), screenshot_som_file, self.font_path
            )
        else:
            output_image_path = screenshot_file

        # Build perception information
        mark_number = 0
        perception_infos = []

        # Add OCR text information
        if self.use_ocr:
            for i in range(len(text_coordinates)):
                mark_number += 1
                if self.use_som and self.draw_text_box:
                    perception_info = {
                        "text": f"mark number: {mark_number} text: {text[i]}",
                        "coordinates": text_coordinates[i],
                    }
                else:
                    perception_info = {"text": f"text: {text[i]}", "coordinates": text_coordinates[i]}
                perception_infos.append(perception_info)

        # Add icon information
        if self.use_icon_detect:
            for i in range(len(icon_coordinates)):
                mark_number += 1
                if self.use_som:
                    perception_info = {"text": f"mark number: {mark_number} icon", "coordinates": icon_coordinates[i]}
                else:
                    perception_info = {"text": "icon", "coordinates": icon_coordinates[i]}
                perception_infos.append(perception_info)

        # Icon description
        if self.use_icon_detect and self.use_icon_caption:
            icon_indices = [i for i in range(len(perception_infos)) if "icon" in perception_infos[i]["text"]]
            if icon_indices:
                icon_boxes = [perception_infos[i]["coordinates"] for i in icon_indices]
                descriptions = await self.icon_tool.caption(screenshot_file, icon_boxes, platform=self.platform)

                # Add description to perception information
                for idx, desc_idx in enumerate(icon_indices):
                    if descriptions.get(idx + 1):
                        perception_infos[desc_idx]["text"] += ": " + descriptions[idx + 1].replace("\n", " ")

        # According to parameter modify coordinate information
        if self.location_info == "center":
            for i in range(len(perception_infos)):
                x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                perception_infos[i]["coordinates"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        elif self.location_info == "icon_center":
            for i in range(len(perception_infos)):
                if "icon" in perception_infos[i]["text"]:
                    x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                    perception_infos[i]["coordinates"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

        # If extend_xml_infos is enabled, then get XML information
        if self.extend_xml_infos and self.platform in ["Android", "Windows"]:
            xml_results = self.controller.get_screen_xml(self.location_info)
            logger.debug(xml_results)
            perception_infos.extend(xml_results)

        return perception_infos, width, height, output_image_path

    def get_webbrowser_console_logs(self, steps: int = 100, expand: bool = True) -> List[Any]:
        """
        Get recent web browser console logs.
        Note: Only used for mgx automated web testing.
        Args:
            steps (int, optional): Number of logs to get, default is 1.
            expand (bool, optional): Whether to return expanded log list, default is True.
                If True, returns the most recent `steps` log list.
                If False, returns the most recent `steps` log dictionary list, containing corresponding operations and console output.
        Returns:
            list: Recent console log list or dictionary list.
        """
        if not self.rc.webbrowser_console_logs:
            return []  # If there is no log, directly return empty list
        if expand:
            return [log for log in self.rc.webbrowser_console_logs[-steps:] if log]  # Filter empty list
        else:
            # Use zip to pair operation history and log
            outputs = [
                {"action": action, "console_output": log}
                for action, log in zip(self.rc.summary_history, self.rc.webbrowser_console_logs)
                if log  # Filter empty list
            ]
            return outputs[-steps:]

    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get action history, including thoughts, summaries, actions, optional memories and reflections.
        Returns:
            list: A list of dictionaries, each dictionary represents a historical record of an action step.
                  Each dictionary contains "thought", "summary", "action",
                  and optional "memory" and "reflection".
        """
        outputs = []
        # Use zip to pair corresponding elements of three historical lists and use enumerate to get index
        for i, (thought, summary, action) in enumerate(
            zip(self.rc.thought_history, self.rc.summary_history, self.rc.action_history)
        ):
            output = {
                "thought": thought,
                "summary": summary,
                "action": action,
            }  # Current step thought  # Current step summary  # Current step action
            # If memory switch is enabled, add memory information
            if self.use_memory:
                output["memory"] = self.rc.memory[i]
            # If reflection switch is enabled, add reflection information
            if self.use_reflection:
                output["reflection"] = self.rc.reflection_thought_history[i]
            outputs.append(output)  # Add current step information to output list
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Information extraction failed, {retry_state.attempt_number}th retry: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _async_memory_task(self, insight: str, screenshot_file: str) -> str:
        """Execute information extraction task asynchronously

        Args:
            insight (str): Content/task description to focus on
            screenshot_file (str): Screenshot file path

        Returns:
            str: Extracted important content
        """
        if not self.use_memory:
            return ""

        return await self.info_extractor.run(insight, screenshot_file)
    

    async def _get_app_package_name(self, app_name: str) -> str:
        """Get application package name

        Args:
            app_name (str): Application name

        Returns:
            str: Application package name
        """
        package_list = self.controller.get_all_packages()

        # Read application mapping information
        map_path = Path(self.knowledge_base_path) / "app_mapping.json"
        app_mapping = ""
        if map_path.exists():
            app_mapping = map_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"{map_path} file does not exist, using default empty mapping")

        # Get package name
        prompt_package_name = self.prompt_utils.get_package_name_prompt(
            app_name=app_name, app_mapping=app_mapping, package_list=package_list
        )

        package_name = await self.llm.aask(
            prompt_package_name,
            system_msgs=[
                f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant."
            ],
            stream=False,
        )

        return package_name.strip()

    async def _handle_open_app(self) -> None:
        """Handle open application action"""
        if self.platform == "Android":
            app_name = re.search(r"\((.*?)\)", self.rc.action).group(1)
            logger.debug(f"Opening Android app: {app_name}")

            package_name = await self._get_app_package_name(app_name)

            if not self.controller.open_app(package_name):
                self.rc.error_flag = True
                logger.error("Failed to start app via adb")
            else:
                time.sleep(10)

        elif self.platform == "Windows":
            app_name = self.rc.action.split("(")[-1].split(")")[0]
            logger.debug(f"Opening Windows app: {app_name}")
            self.controller.open_app(app_name)
            time.sleep(10)
        else:
            logger.error(f"Platform {self.platform} not supported for opening apps")

    async def _act(self) -> bool:
        """Execute action step"""
        if self.use_chrome_debugger:
            # Store browser logs from before action execution in previous action log. Note: Need a log for step 0 here since mgx web testing is not started by osagent
            self.rc.webbrowser_console_logs.append(self.chrome_debugger.get_new_messages())

        self.run_action_failed = False
        self.run_action_failed_exception = ""

        # Execute action
        if "Stop" in self.rc.action:
            # If it's a stop operation, end the loop
            return False
        elif "Open App" in self.rc.action:
            await self._handle_open_app()
        else:
            # Execute other actions
            try:
                if self.platform in ["Android", "Windows"]:
                    self.controller.run_action(self.rc.action)
                else:
                    logger.error("Currently only supports Android and Windows")
            except Exception as e:
                # For direct exit when using tell in automg
                if isinstance(e, SystemExit) and e.code == 0:
                    return False
                logger.error(f"run action failed: {e}")
                self.run_action_failed = True
                self.run_action_failed_exception = e
                return False

        time.sleep(0.5)

        return True

    def _deduplicate_by_coordinates(self, infos: List[Dict]) -> List[Dict]:
        """Remove duplicate infos by coordinates"""
        seen = set()
        unique = []
        for info in infos:
            coord_key = tuple(info["coordinates"]) if isinstance(info["coordinates"], list) else info["coordinates"]
            if coord_key not in seen:
                seen.add(coord_key)
                unique.append(info)
        return unique

    async def _filter_perception_infos(self, perception_infos: List[Dict]) -> List[Dict]:
        """Filter perception infos"""
        
        # Call LLM to generate decision
        images = [encode_image(self.screenshot_file)]
        if self.use_som:
            images.append(encode_image(self.screenshot_som_file))

        # Use custom system prompt or default prompt
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant. You need to help me operate the device to complete the user's instruction."
        )

        perception_infos_str = "\n".join([f"{i+1}. {{coordinates: {info['coordinates']}, description: {{{info['text']}}}}}" for i, info in enumerate(perception_infos)])
        filter_prompt = f"""
Please filter the top 10 important infos or infos that can be clicked to complete a task from the following perception infos:
{perception_infos_str}

Return the filtered infos in the following format without any other text:
[
    {{"coordinates": (x, y), "description": "text content"}},
    ...
]
"""
        try:
            output = await self.llm.aask(
                filter_prompt,
                system_msgs=[system_msg],
                images=images,
                stream=False,
            )
            
            # Try to parse JSON output
            import json
            import re
            
            # extract 10 texts
            texts = re.findall(r'"text": "(.*?)",', output)
            coordinates = re.findall(r'"coordinates": \((.*?), (.*?)\)', output)
            perception_infos_filtered = [{"text": text, "coordinates": (int(x), int(y))} for text, (x, y) in zip(texts, coordinates)]
                
        except Exception as e:
            logger.error(f"Error in filtering perception infos: {e}")
            return perception_infos
    def calculate_dhash(self, image_pil, hash_size=8):
        """
        Calculates the Difference Hash (dHash) of a PIL Image.

        Args:
            image_pil (PIL.Image.Image): The input image.
            hash_size (int): The size of the hash (e.g., 8 means 8x8 = 64-bit hash).

        Returns:
            str: The dHash string.
        """
        # 1. Resize to (hash_size + 1, hash_size) - one extra column for comparison
        # The comparison is row-wise, so we need hash_size rows and hash_size+1 columns
        # to get hash_size comparisons per row.
        resized_image = image_pil.resize((hash_size + 1, hash_size), Image.LANCZOS)
        
        # 2. Convert to grayscale
        grayscale_image = resized_image.convert('L')
        
        # 3. Calculate differences and construct hash
        dhash = []
        for y in range(hash_size): # Iterate through rows
            for x in range(hash_size): # Iterate through columns for comparison
                pixel_left = grayscale_image.getpixel((x, y))
                pixel_right = grayscale_image.getpixel((x + 1, y))
                if pixel_left > pixel_right:
                    dhash.append('1')
                else:
                    dhash.append('0')
        return "".join(dhash)

    def compare_hashes_similarity(self, hash1, hash2):
        """
        Compares two dHashes and returns a similarity score.

        Args:
            hash1 (str): The first dHash string.
            hash2 (str): The second dHash string.

        Returns:
            float: Similarity score between 0.0 (completely different) and 
                1.0 (identical). Returns -1.0 if hash lengths differ.
        """
        if len(hash1) != len(hash2):
            return -1.0 # Hashes must be same length to compare

        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        hash_length = len(hash1)
        
        similarity = (hash_length - hamming_distance) / hash_length
        return similarity
    
    def _update_classified_perception_infos(self, last: Dict, now: Dict) -> bool:
        # screenshot similarity
        last_screenshot = Image.open(last["screenshot_file"])
        now_screenshot = Image.open(now["screenshot_file"])
        # last_screenshot_array = np.array(last_screenshot)
        # now_screenshot_array = np.array(now_screenshot)
        # screenshot_similarity = np.mean(last_screenshot_array == now_screenshot_array)
        hash_last = self.calculate_dhash(last_screenshot)
        hash_now = self.calculate_dhash(now_screenshot)
        screenshot_similarity = self.compare_hashes_similarity(hash_last, hash_now)
        breakpoint()
        
        # perception infos similarity
        last_perception_infos = last["perception_infos"]
        now_perception_infos = now["perception_infos"]
        last_coordinates = [tuple(info["coordinates"]) for info in last_perception_infos]
        now_coordinates = [tuple(info["coordinates"]) for info in now_perception_infos]
        last_coordinates_set = set(last_coordinates)
        now_coordinates_set = set(now_coordinates)
        coordinates_diff = len(last_coordinates_set - now_coordinates_set) / len(last_coordinates_set)

        if screenshot_similarity < 0.9 or coordinates_diff > 0.1:
            return True
        return False
    
    async def _classify_perception_infos(self, perception_infos: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify perception infos into different categories"""
        # Call LLM to generate decision
        images = [encode_image(self.screenshot_file)]
        if self.use_som:
            images.append(encode_image(self.screenshot_som_file))

        system_msg = f"You are an expert UI element classification assistant for {'mobile phone' if self.platform=='Android' else 'PC'} interfaces. Your task is to analyze and categorize UI elements from screenshots based on their visual appearance, text content, and functional purpose. You should group similar elements together and provide meaningful category names that reflect their functionality or type (e.g., 'Navigation Buttons', 'Text Fields', 'Menu Items', 'Content Areas', 'Action Buttons', 'Status Indicators', etc.). Focus on creating logical groupings that would be useful for automated interaction."

        perception_infos_unique = self._deduplicate_by_coordinates(perception_infos)
        
        perception_infos_str = "\n".join([f"{i+1}. {{coordinates: {info['coordinates']}, description: {{{info['text']}}}}}" for i, info in enumerate(perception_infos_unique)])
        rm_description_prompt = False
        info_prompt = r'{"coordinates": (x, y)}' if rm_description_prompt else r'{"coordinates": (x, y), "description": "text content"}'

        classify_prompt = f"""
Please classify the following perception infos into different categories:
{perception_infos_str}

You can filter out some infos that are window control info and navigation bar control info, but other infos should be all classified, don't miss any infos!

Return the classification result in the following format without any other text:
[
    {{"category": "category name", "infos": [{info_prompt}, ...]}},
    ...
]
"""
        output = await self.llm.aask(
                classify_prompt,
                system_msgs=[system_msg],
                images=images,
                stream=False,
            )

        categories = re.findall(r'"category": "(.*?)",', output)
        classification_result = [
            {"category": category, "infos": []}
            for category in categories
        ]
        infos = output.split("infos")
        idx = 0
        for info in infos:
            if "coordinates" in info:
                coordinates = re.findall(r'"coordinates": \((.*?), (.*?)\)', info)
                if rm_description_prompt:
                    descriptions = [""] * len(coordinates)
                else:
                    descriptions = re.findall(r'"description": "(.*?)"', info)
                classification_result[idx]["infos"] = [{"coordinates": (x, y), "text": description} for description, (x, y) in zip(descriptions, coordinates)]
                idx += 1
        logger.info(f"Filtered infos {sum([len(info['infos']) for info in classification_result])} / {len(perception_infos_unique)}")
        return classification_result
    
    def visual_classification(self, classified_infos: List[Dict], image_path: str) -> None:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        text_width = 10
        text_height = 10
        for info in classified_infos:
            # color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for item in info["infos"]:
                x = int(item["coordinates"][0])
                y = int(item["coordinates"][1])
                draw.rectangle([x, y, x + 20, y + 20], outline=color, width=2, fill=color)

            draw.text((text_width, text_height), f"{info['category']}", fill=color, stroke_width=2, stroke_fill=color)
            text_height += 20
        image.save(image_path.replace(".jpg", f"_classified_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))

    async def _dfs(self, name="", max_depth=5, max_width=3) -> None:
        """Depth-first search (DFS) for exploring clickable elements on the screen.
        Args:
            name (str): Name of the current DFS iteration.
        """
        last_iter_info = {
            "perception_infos": self.rc.perception_infos,
            "screenshot_file": self.rc.screenshot_origin_file,
        }
        
        self._update_screenshot_files()

        # Get new perception information
        self.rc.perception_infos, self.width, self.height, self.output_image_path = await self._get_perception_infos(
            self.screenshot_file, self.screenshot_som_file
        )
        # Save images
        self._save_iteration_images(name)
        
        current_iter_info = {
            "perception_infos": self.rc.perception_infos,
            "screenshot_file": self.rc.screenshot_origin_file,
        }

        if len(self.rc.classified_perception_infos) == 0 or self._update_classified_perception_infos(last_iter_info, current_iter_info):
            self.rc.classified_perception_infos = await self._classify_perception_infos(self.rc.perception_infos)
            self.visual_classification(self.rc.classified_perception_infos, self.screenshot_file)

        self.rc.next_perception_infos = [] # clear next_perception_infos, just save this iter

        t_max_width = min(max_width, len(self.rc.classified_perception_infos))
        # Randomly sample classified_infos up to t_max_width
        sampled_classified_infos = random.sample(self.rc.classified_perception_infos, t_max_width)
        
        for classified_info in sampled_classified_infos:
            info = classified_info["infos"].pop(0)
            self.rc.next_perception_infos.append({"category": classified_info["category"], "info": info})
        
        # Remove all empty classified_infos from the original list after processing
        self.rc.classified_perception_infos = [
            item for item in self.rc.classified_perception_infos if len(item["infos"]) > 0
        ]
            
        # Save current state
        self.rc.save_state(path=self.state_dir / f"{name}.json")
        
        if len(name.split("_")) > max_depth or not self.rc.completion_status:
            return
        
        # random sample 5 coordinates
        # sample_num = min(2, len(perception_infos_unique))
        # if len(perception_infos_unique) > sample_num:
        #     perception_infos_unique = dict(random.sample(list(perception_infos_unique.items()), sample_num))
        
        for idx, info in enumerate(self.rc.next_perception_infos):
            coordinates = info["info"]["coordinates"]
            action = f"Run (pyautogui.click({coordinates[0]}, {coordinates[1]}); time.sleep(1);)"
            self.rc.action = action
            self.rc.completion_status = await self._act()
            self.rc.action_history.append(self.rc.action)
            await self._dfs(f"{name}_{idx}", max_depth=max_depth, max_width=max_width)
            self.rc.load_state(path=self.state_dir / f"{name}.json")
            await self._replay()
    
    def _maximize_window(self) -> None:
        windows = Desktop(backend="uia").windows()
        for w in windows:
            try:
                # 检查窗口是否可见且支持最大化
                if w.is_visible() and hasattr(w, 'maximize'):
                    # 过滤掉系统窗口和特殊窗口
                    window_text = w.window_text() if hasattr(w, 'window_text') else ""
                    class_name = w.class_name() if hasattr(w, 'class_name') else ""
                    
                    # 跳过系统窗口、任务栏、通知等
                    skip_windows = [
                        "Taskbar", "任务栏", "Start", "开始菜单", 
                        "Notification", "通知", "Desktop", "桌面",
                        "Shell_TrayWnd", "NotifyIconOverflowWindow"
                    ]
                    
                    if any(skip in window_text for skip in skip_windows) or \
                       any(skip in class_name for skip in skip_windows):
                        logger.info(f"Skipping system window: {window_text} ({class_name})")
                        continue
                    
                    # 检查窗口是否已经最大化
                    if not w.is_maximized():
                        w.maximize()
                        logger.info(f"Maximized window: {window_text} ({class_name})")
                    else:
                        logger.info(f"Window already maximized: {window_text} ({class_name})")
            except Exception as e:
                logger.warning(f"Failed to maximize window {w}: {e}")
        
    async def _replay(self) -> Message:
        windows = Desktop(backend="uia").windows()
        for w in windows:
            try:
                w.close()
            except Exception as e:
                logger.warning(f"Failed to close window {w}: {e}")
        time.sleep(1)
        for action in self.rc.init_action_list:
            self.rc.action = action
            logger.info(f"Initial action: {self.rc.action}")
            await self._act()
        
        self._maximize_window()
   
        for action in self.rc.action_history:
            self.rc.action = action
            logger.info(f"Replay action: {self.rc.action}")
            await self._act()
        logger.info(f"Replay finished, last action: {self.rc.action}")

    async def _react(self) -> Message:
        await self._replay()
        rsp = AIMessage(content="No actions taken yet", cause_by=Action)  # will be overwritten after Role _act
        (
            self.rc.perception_infos,
            self.width,
            self.height,
            self.output_image_path,
        ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

        await self._dfs(name="0")
        
        if self.use_chrome_debugger:
            self.chrome_debugger.stop_monitoring()

        return rsp
    
    async def _collect_info(self, last_file_name: str, file_path_list: List[Path]) -> Tuple[List[Dict], List[str], List[str], List[str], List[str]]:
        perception_infos_list = []
        screenshot_origin_file_list = []
        screenshot_draw_file_list = []
        action_history = []
        state_list = []
        gen_instruction_list = []
        
        for file_path in file_path_list:
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            state_list.append(state)
            perception_infos_list.append(state["perception_infos"])
            screenshot_origin_file_list.append(state["screenshot_origin_file"])
            screenshot_draw_file_list.append(state["screenshot_draw_file"])
            if "gen_instruction" in state:
                gen_instruction_list.append(state["gen_instruction"][last_file_name])
            
            action_history = state["action_history"]
        assert len(perception_infos_list) == len(screenshot_origin_file_list) == len(screenshot_draw_file_list) == len(action_history) + 1
        return state_list, perception_infos_list, screenshot_origin_file_list, action_history, gen_instruction_list
    
    async def _back_generate(self, path: Optional[Path | str] = None) -> None:
        if isinstance(path, str):
            path = Path(path)
        file_dir = path.parent
        gen_dir = file_dir.parent / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        file_name = path.stem
        file_name_list = [f"{file_name[:i + 1]}" for i in range(0, len(file_name), 2)]
        file_path_list = [file_dir / f"{file_name}.json" for file_name in file_name_list]
        
        (state_list, perception_infos_list, screenshot_origin_file_list, action_history, _) = await self._collect_info(file_name, file_path_list)
        if len(perception_infos_list) == 0:
            return
        images = []
        for i in range(len(perception_infos_list)):
            images.append(encode_image(screenshot_origin_file_list[i]))


        iterations_info = ""
        for i in range(len(perception_infos_list)):
            iterations_info += f"""### Info of the current iteration {i + 1} ###
Perception info {i + 1}:
{perception_infos_list[i]}
<Screenshot {i + 1}>
"""
            if i < len(action_history):
                iterations_info += f"""Action {i + 1}:
{action_history[i]}
"""
            else:
                iterations_info += "Final state (no action)\n"
            
            iterations_info += f"### Iteration {i + 1} end ###\n"

        step_prompt = """Step 1:
<think>
a natural and concise thinking process (150-300 words) explaining its visual analysis, intent interpretation, and reasoning for interacting with specific UI elements to achieve the goal
</think>
<summary>
one sentence summary of the current step's operation to take towards the goal
</summary>
<tasklist>
* **[Completed Tasks]:** (List the tasks that have been successfully completed so far)
    * <Task 1 Description>
    * <Task 2 Description>
    ...
* **[Current Task]:** <Current Task Description>.
* **[Next Operation]:** (Describe the immediate next operation in detail, including what needs to be done)
    * <Operation 1 Description>
    * <Operation 2 Description>
    ...
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
    * <Task 1 Description>
    * <Task 2 Description>
</tasklist>
"""
        for i in range(len(action_history) - 1):
            step_prompt += f"""Step {i + 2}:
...
"""

        gen_prompt = f"""Please generate a task instruction and step-by-step task list, thinking process and operation for each action taken to complete the task from the following iteration infos:
###### Iteration Infos ######
{iterations_info}
###### Iteration Infos End ######


Please provide:
1. A clear and general task instruction that describes what the user wants to accomplish
2. A planned task list that need to be completed to achieve the goal, based on the task instruction and the action history
3. Step-by-step task list, thinking process and operation for each action taken to complete the task

Note:
1. The total number of steps should be equal to the number of actions taken.
2. One task can be completed in multiple steps.
3. The summary of each step should keep consistent with the action history and the current task.


Return the result in the following format without any other text:
### Task Instruction ###
a clear and general task instruction

### Initial Task List ###
* **[Completed Tasks]:** 
  * None
* **[Current Task]:** <describe the first high-level task to execute>
* **[Next Operation]:** 
  * <describe the first operation in detail>
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
  * <describe remaining high-level task 1>
  * <describe remaining high-level task 2>
  * ...

### Step-by-Step Execution Process ### (total {len(action_history)} steps!!!)
{step_prompt}

### End ###
"""
        # with open("tmp.txt", "w", encoding="utf-8") as f:
        #     f.write(gen_prompt)
        # breakpoint()
        
        system_msg = "You are a helpful AI assistant for generating task instructions and step-by-step task list, thinking process and operations for each action taken to complete user tasks. You excel at analyzing screenshots and user interactions to create detailed, structured training data for AI agents."
        
        output = await self.llm.aask(
            gen_prompt,
            system_msgs=[system_msg],
            images=images,
            stream=False,
        )
        
        # with open("output.txt", "w", encoding="utf-8") as f:
        #     f.write(output)
        # breakpoint()

        task_instruction = re.findall(r"### Task Instruction ###(.*?)###", output, re.DOTALL)
        task_instruction = task_instruction[0].strip() if task_instruction else ""
        
        # Extract task list
        init_task_list = re.findall(r"### Initial Task List ###\s*(.*?)\s*###", output, re.DOTALL)
        init_task_list = init_task_list[0].strip() if init_task_list else ""
        
        # Extract step instructions
        steps_section = re.findall(r"### Step-by-Step Execution Process ###\s*(.*?)\s*### End ###", output, re.DOTALL)
        steps_content = steps_section[0] if steps_section else ""
        step_pattern = r"Step (\d+):\s*<think>(.*?)</think>\s*<summary>(.*?)</summary>\s*<tasklist>(.*?)</tasklist>"
        step_matches = re.findall(step_pattern, steps_content, re.DOTALL)
    
        step_instructions = []
        for step_num, think_content, operation_content, tasklist_content in step_matches:
            step_instructions.append({
                "step": step_num,
                "tasklist": tasklist_content.strip(),
                "think": think_content.strip(),
                "operation": operation_content.strip()
            })

        for i in range(len(state_list)):
            output_path = gen_dir / f"{file_name_list[i]}.json"
            if output_path.exists():
                state = json.load(output_path.open("r", encoding="utf-8"))
            else:
                state = state_list[i]
            if "gen_instruction" not in state:
                state["gen_instruction"] = {}
            state["gen_instruction"][file_name_list[-1]] = {
                "task_instruction": task_instruction,
                "init_tasklist": init_task_list,
                "step_instructions": step_instructions
            }
            with open(gen_dir / f"{file_name_list[i]}.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
                
    def _conv_format(self, sys_msg: str, user_msg: List[str], assistant_msg: List[str], images: List[str]=[]) -> List[Dict]:
        assert len(user_msg) == len(assistant_msg)
        message = []
        message.append({"role": "system", "content": sys_msg})
        for i in range(len(user_msg)):
            message.append({"role": "user", "content": f"<image>{user_msg[i]}"})
            message.append({"role": "assistant", "content": assistant_msg[i]})
        return {
            "message": message,
            "images": images
        }
    
    async def _gen_action_prompt(self, path) -> str:
        if isinstance(path, str):
            path = Path(path)
        file_dir = path.parent
        gen_dir = file_dir.parent / "gen"
        prompt_dir = file_dir.parent / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        file_name = path.stem
        file_name_list = [f"{file_name[:i + 1]}" for i in range(0, len(file_name), 2)]
        file_path_list = [gen_dir / f"{file_name}.json" for file_name in file_name_list]
        (state_list, perception_infos_list, screenshot_origin_file_list, action_history, gen_instruction_list) = await self._collect_info(file_name, file_path_list)
        
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant. You need to help me operate the device to complete the user's instruction."
        )
        
        ans_template = """### Thought ###
{think}
### Action ###
{action}

### Operation ###
{operation}

### Task List ###
{tasklist}
"""

        last_perception_infos = perception_infos_list[0]
        last_task_list = None
        
        instruction = gen_instruction_list[0]["task_instruction"]
        initial_task_prompt = f"""Based on the following instruction, please generate an initial task list:
{instruction}

Please output the task list in the following format:
* **[Completed Tasks]:** 
  * None
* **[Current Task]:** <describe the first high-level task to execute>
* **[Next Operation]:** 
  * <describe the first step in detail>
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
  * <describe remaining high-level task 1>
  * <describe remaining high-level task 2>
  * ...
"""
        initial_task_list = gen_instruction_list[0]["init_tasklist"]
        last_task_list = initial_task_list

        initial_task_data = {
            "message": self._conv_format(system_msg, [initial_task_prompt], [initial_task_list]),
            "images": [screenshot_origin_file_list[0]]
        }
        with open(prompt_dir / f"{file_name}_init_tasklist.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(initial_task_data, ensure_ascii=False) + "\n")
        
        user_msg = []
        user_msg_perception = []
        assistant_msg = []
        images = []
        
        for i in range(1, len(state_list)):
            state = state_list[i]
            perception_infos = perception_infos_list[i]
            gen_instruction = gen_instruction_list[i]
            instruction = gen_instruction["task_instruction"]
            step_instructions = gen_instruction["step_instructions"]
                    
            action_history = state["action_history"]
            thought_history = [step_instruction["think"] for step_instruction in step_instructions]
            summary_history = [step_instruction["operation"] for step_instruction in step_instructions]
                    
            # generate ans
            ans_think = thought_history[-1]  # last of thought_history
            ans_operation = summary_history[-1]  # last of summary_history
            ans_action = action_history[-1]
            ans_task_list = step_instructions[-1]["tasklist"]
            ans_action = ans_template.format(
                think=ans_think,
                action=ans_action,
                operation=ans_operation,
                tasklist=ans_task_list
            )
                    
            thought_history = thought_history[:-1]
            summary_history = summary_history[:-1]
            action_history = action_history[:-1]
             
            # Generate action
            ctx = ActionPromptContext(
                instruction=instruction,
                clickable_infos=last_perception_infos,
                width=1920,
                height=1080,
                thought_history=thought_history,
                summary_history=summary_history,
                action_history=action_history,
                reflection_thought_history=[],
                last_summary=summary_history[-1] if len(summary_history) > 0 else "",
                last_action=action_history[-1] if len(action_history) > 0 else "",
                reflection_thought="",
                add_info="",
                error_flag=False,
                completed_content="",
                memory=[],
                task_list=last_task_list,
                use_som=False,
                location_info="center",
            )            
            user_msg_perception.append(self.prompt_utils.get_action_prompt(ctx))
            ctx.clickable_infos = []
            user_msg.append(self.prompt_utils.get_action_prompt(ctx))
            assistant_msg.append(ans_action)
            images.append(screenshot_origin_file_list[i - 1])
            
            last_perception_infos = perception_infos
            last_task_list = ans_task_list
            
        assert len(user_msg) == len(assistant_msg) == len(images)
        
        with open(prompt_dir / f"{file_name}_onestep_perception.jsonl", "w", encoding="utf-8") as f:
            for u, a, i in zip(user_msg_perception, assistant_msg, images):
                f.write(json.dumps(self._conv_format(system_msg, [u], [a], [i]), ensure_ascii=False) + "\n")
            
        with open(prompt_dir / f"{file_name}_onestep.jsonl", "w", encoding="utf-8") as f:
            for u, a, i in zip(user_msg, assistant_msg, images):
                f.write(json.dumps(self._conv_format(system_msg, [u], [a], [i]), ensure_ascii=False) + "\n")

        with open(prompt_dir / f"{file_name}_all_perception.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(self._conv_format(system_msg, user_msg_perception, assistant_msg, images), ensure_ascii=False))
        
        with open(prompt_dir / f"{file_name}_all.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(self._conv_format(system_msg, user_msg, assistant_msg, images), ensure_ascii=False))
        

    async def run(self, instruction: str) -> Message:
        """Run main loop.

        Args:
            instruction (str): User instruction.
        """
        self._reset_state()  # Reset state for each run
        self._setup_logs()  # Reset logs for each run
        self.instruction = instruction
        
        # await self.full_replay("workspace/202505282027/state/0_1_0_0_1.json")
        # await self.full_replay("workspace/202505301539/state/0_0_0_0_0_0_0.json")
        
        rsp = await self._react()
        
        file_dir = Path("workspace/202506040936/state")
        # max length filename
        max_filename_length = max([len(file.stem) for file in file_dir.glob("*.json")])
        file_path_list = [file for file in file_dir.glob("*.json") if len(file.stem) == max_filename_length]
        # breakpoint()
        # for file_path in file_path_list:
        #     rsp = await self._back_generate(file_path)

        for file_path in file_path_list:
            rsp = await self._gen_action_prompt(file_path)
        
        return ""
    
    async def full_replay(self, filepath: str) -> None:
        self.rc.load_state(path=filepath)
        await self._replay()
