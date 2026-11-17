from .nodes import MagikLatentMerge, MagikLatentSplit

NODE_CLASS_MAPPINGS = {
    "MagikLatentSplit": MagikLatentSplit,
    "MagikLatentMerge": MagikLatentMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagikLatentSplit": "Magik Latent Split",
    "MagikLatentMerge": "Magik Latent Merge",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
