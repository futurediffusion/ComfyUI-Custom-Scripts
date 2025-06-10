import os
import glob
import random
import json

USER_DIR = os.path.abspath(os.path.join(__file__, "../../user"))
STATE_FILE = os.path.join(USER_DIR, "prompt_folder_state.json")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


_state = load_state()


def _read_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]

class PromptFolder:
    """Iterate prompts from .txt files in a folder"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "prompts"}),
                "order": (["ordered", "random"], {}),
                "index": ("INT", {"default": 0, "min": 0}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "reset"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_prompt"
    CATEGORY = "utils"

    def get_prompt(self, folder, order, index, reset):
        folder = os.path.abspath(folder)
        state = _state.get(folder)
        if state is None:
            prompts = []
            for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                prompts.extend(_read_prompts(path))
            if order == "random":
                random.shuffle(prompts)
            start = max(0, min(index, len(prompts)))
            state = {"prompts": prompts, "index": start, "order": order}
            _state[folder] = state
        else:
            if reset:
                start = max(0, min(index, len(state["prompts"])))
                state["index"] = start
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
        save_state(_state)
        return (prompt,)


NODE_CLASS_MAPPINGS = {"PromptFolder|pysssss": PromptFolder}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptFolder|pysssss": "Prompt Folder 🐍"}
