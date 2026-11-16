class MagikLatentSplit:
    SPLIT_COUNT = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",) * SPLIT_COUNT
    RETURN_NAMES = tuple(f"latent_{i + 1}" for i in range(SPLIT_COUNT))
    FUNCTION = "split"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Splits a latent into 4 equal latents. Video latents [B, C, T, H, W] are "
        "split along the frame axis; image latents [B, C, H, W] along the batch axis."
    )

    def split(self, latent):
        if "samples" not in latent:
            raise ValueError("MagikLatentSplit: input latent has no 'samples' tensor")
        samples = latent["samples"]
        ndim = len(samples.shape)
        if ndim == 5:
            dim = 2
            axis = "frames"
        elif ndim == 4:
            dim = 0
            axis = "batch"
        else:
            raise ValueError(
                f"MagikLatentSplit: expected a 4D image latent [B, C, H, W] or a 5D "
                f"video latent [B, C, T, H, W], got {ndim}D samples with shape "
                f"{tuple(samples.shape)}"
            )
        size = samples.shape[dim]
        if size % self.SPLIT_COUNT != 0:
            raise ValueError(
                f"MagikLatentSplit: cannot split {size} {axis} into "
                f"{self.SPLIT_COUNT} equal parts (samples shape {tuple(samples.shape)})"
            )
        step = size // self.SPLIT_COUNT
        outputs = []
        for i in range(self.SPLIT_COUNT):
            index = [slice(None)] * ndim
            index[dim] = slice(i * step, (i + 1) * step)
            out = dict(latent)
            out["samples"] = samples[tuple(index)]
            outputs.append(out)
        return tuple(outputs)
