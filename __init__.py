from .nodes import (
    MagikConcatAVLatent,
    MagikLatentMerge,
    MagikLatentSplit,
    MagikLatentSplitAdvanced,
    MagikSeparateAVLatent,
)

NODE_CLASS_MAPPINGS = {
    "MagikLatentSplit": MagikLatentSplit,
    "MagikLatentSplitAdvanced": MagikLatentSplitAdvanced,
    "MagikLatentMerge": MagikLatentMerge,
    "MagikSeparateAVLatent": MagikSeparateAVLatent,
    "MagikConcatAVLatent": MagikConcatAVLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagikLatentSplit": "Magik Latent Split",
    "MagikLatentSplitAdvanced": "Magik Latent Split Advanced",
    "MagikLatentMerge": "Magik Latent Merge",
    "MagikSeparateAVLatent": "Magik Separate AV Latent",
    "MagikConcatAVLatent": "Magik Concat AV Latent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
