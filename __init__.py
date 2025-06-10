from .py.prompt_folder import PromptFolder
from .py.prompt_folder_advanced import PromptFolderAdvanced

NODE_CLASS_MAPPINGS = {
    "PromptFolder": PromptFolder,
    "PromptFolderAdvanced": PromptFolderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptFolder": "Prompt Folder",
    "PromptFolderAdvanced": "Prompt Folder Advanced",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
