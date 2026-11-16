class MagikLatentSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",) * 4
    RETURN_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
    FUNCTION = "split"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Splits a latent into 4 spatial quadrants (2x2 grid): top-left, top-right, "
        "bottom-left, bottom-right. Supports image latents [B, C, H, W] and video "
        "latents [B, C, T, H, W]; H and W must be even."
    )

    def split(self, latent):
        if "samples" not in latent:
            raise ValueError("MagikLatentSplit: input latent has no 'samples' tensor")
        samples = latent["samples"]
        ndim = len(samples.shape)
        if ndim == 5:
            h_dim, w_dim = 3, 4
        elif ndim == 4:
            h_dim, w_dim = 2, 3
        else:
            raise ValueError(
                f"MagikLatentSplit: expected a 4D image latent [B, C, H, W] or a 5D "
                f"video latent [B, C, T, H, W], got {ndim}D samples with shape "
                f"{tuple(samples.shape)}"
            )
        h, w = samples.shape[h_dim], samples.shape[w_dim]
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"MagikLatentSplit: cannot split H={h}, W={w} into 4 equal quadrants "
                f"(both must be even; samples shape {tuple(samples.shape)})"
            )
        h_step, w_step = h // 2, w // 2
        outputs = []
        for row in range(2):
            for col in range(2):
                index = [slice(None)] * ndim
                index[h_dim] = slice(row * h_step, (row + 1) * h_step)
                index[w_dim] = slice(col * w_step, (col + 1) * w_step)
                out = dict(latent)
                out["samples"] = samples[tuple(index)]
                outputs.append(out)
        return tuple(outputs)
