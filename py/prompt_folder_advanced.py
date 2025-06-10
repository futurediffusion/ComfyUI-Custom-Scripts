import os
import glob
import random

class PromptFolderAdvanced:
    def __init__(self):
        self.current_index = 0
        self.prompts = []
        self.total_prompts = 0
        self.current_folder = ""
        self.random_indices = set()
        self.last_traversal_mode = "forward"
        self.last_non_held_index = None
        self.held_index = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "traversal_mode": (["forward", "reverse", "random"], {"default": "forward"}),
                "skip_lines": ("INT", {"default": 0, "min": 0, "max": 10}),
                "reset_counter": ("BOOLEAN", {"default": False}),
                "reload_folder": ("BOOLEAN", {"default": False}),
                "hold_current_text": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "starting_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("prompt", "current_index", "total_prompts", "remaining_prompts")
    FUNCTION = "process_folder"
    CATEGORY = "utils"

    def should_reload_folder(self, folder_path, reload_folder):
        if reload_folder:
            return True
        return folder_path != self.current_folder

    def load_folder(self, folder_path, traversal_mode):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        prompts = []
        for file in sorted(glob.glob(os.path.join(folder_path, "*.txt"))):
            with open(file, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    line = line.strip()
                    if line:
                        prompts.append(line)

        self.prompts = prompts
        self.total_prompts = len(prompts)
        self.current_folder = folder_path

        if self.total_prompts == 0:
            raise ValueError(f"No prompts found in folder: {folder_path}")

        if traversal_mode == "random":
            self.random_indices = set(range(self.total_prompts))
        else:
            self.random_indices = set()

    def adjust_index_for_mode_change(self, new_mode):
        if new_mode != self.last_traversal_mode:
            if new_mode == "random":
                self.random_indices = set(range(self.total_prompts))
                if self.current_index in self.random_indices:
                    self.random_indices.remove(self.current_index)
            elif new_mode == "reverse" and self.last_traversal_mode == "forward":
                self.current_index = (self.current_index - 1) % self.total_prompts
            elif new_mode == "forward" and self.last_traversal_mode == "reverse":
                self.current_index = (self.current_index + 1) % self.total_prompts
            self.last_traversal_mode = new_mode

    def get_next_index(self, traversal_mode, skip_lines):
        if not self.prompts:
            return 0

        skip_amount = skip_lines + 1

        if traversal_mode == "forward":
            next_idx = self.current_index
            self.current_index = (self.current_index + skip_amount) % self.total_prompts
            return next_idx
        elif traversal_mode == "reverse":
            next_idx = self.current_index
            self.current_index = (self.current_index - skip_amount) % self.total_prompts
            return next_idx
        else:
            if not self.random_indices:
                self.random_indices = set(range(self.total_prompts))
            if not self.random_indices:
                self.random_indices = set(range(self.total_prompts))
            next_idx = random.choice(list(self.random_indices))
            self.random_indices.remove(next_idx)
            for _ in range(skip_lines):
                if self.random_indices:
                    self.random_indices.remove(random.choice(list(self.random_indices)))
            return next_idx

    def get_remaining(self, traversal_mode):
        if not self.prompts:
            return 0
        if traversal_mode == "random":
            return len(self.random_indices)
        elif traversal_mode == "forward":
            return self.total_prompts - self.current_index
        else:
            return self.current_index + 1

    def process_folder(self, folder_path, traversal_mode="forward", skip_lines=0, reset_counter=False, reload_folder=False, hold_current_text=False, starting_index=None):
        if self.should_reload_folder(folder_path, reload_folder):
            self.load_folder(folder_path, traversal_mode)
            self.current_index = starting_index if starting_index is not None else 0
            self.last_traversal_mode = traversal_mode
            self.last_non_held_index = None
            self.held_index = None
        elif reset_counter:
            self.current_index = starting_index if starting_index is not None else 0
            self.last_traversal_mode = traversal_mode
            if traversal_mode == "random":
                self.random_indices = set(range(self.total_prompts))
            self.last_non_held_index = None
            self.held_index = None
        elif starting_index is not None and self.current_index == 0:
            self.current_index = starting_index

        self.adjust_index_for_mode_change(traversal_mode)

        if not self.prompts:
            return ("", 0, 0, 0)

        if hold_current_text:
            if self.held_index is not None:
                idx = self.held_index
            else:
                if self.last_non_held_index is not None:
                    idx = self.last_non_held_index
                else:
                    idx = self.get_next_index(traversal_mode, skip_lines)
                self.held_index = idx
        else:
            idx = self.get_next_index(traversal_mode, skip_lines)
            self.last_non_held_index = idx
            self.held_index = None

        prompt = self.prompts[idx]
        remaining = self.get_remaining(traversal_mode)
        return (prompt, idx + 1, self.total_prompts, remaining)

NODE_CLASS_MAPPINGS = {
    "PromptFolderAdvanced|pysssss": PromptFolderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptFolderAdvanced|pysssss": "Prompt Folder Advanced 🐍",
}

