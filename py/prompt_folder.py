import os
import glob
import random

_state = {}

class PromptFolder:
    """Iterate prompts from .txt files in a folder"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "prompts"}),
                "order": (["ordered", "random"], {}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "reset"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_prompt"
    CATEGORY = "utils"

    def get_prompt(self, folder, order, reset):
        folder = os.path.abspath(folder)
        state = _state.get(folder)
        if state is None:
            prompts = []
            for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f.read().splitlines():
                        line = line.strip()
                        if line:
                            prompts.append(line)
            if order == "random":
                random.shuffle(prompts)
            state = {"prompts": prompts, "index": 0, "order": order}
            _state[folder] = state
        else:
            if reset:
                state["index"] = 0
                if order == "random":
                    random.shuffle(state["prompts"])
            state["order"] = order

        if not state["prompts"]:
            return ("",)

        if state["index"] >= len(state["prompts"]):
            state["index"] = 0
            if state["order"] == "random":
                random.shuffle(state["prompts"])

        prompt = state["prompts"][state["index"]]
        state["index"] += 1
        return (prompt,)


NODE_CLASS_MAPPINGS = {"PromptFolder|pysssss": PromptFolder}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptFolder|pysssss": "Prompt Folder 🐍"}
