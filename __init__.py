from .nodes import MagikLatentSplit

NODE_CLASS_MAPPINGS = {
    "MagikLatentSplit": MagikLatentSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagikLatentSplit": "Magik Latent Split",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
