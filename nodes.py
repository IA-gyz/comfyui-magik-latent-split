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
        "bottom-left, bottom-right. Image latents [B, C, H, W] and video latents "
        "[B, C, T, H, W] are supported; H and W must be even. For nested latents "
        "(e.g. MiniMax H3 AV) the first stream is split and the remaining streams "
        "(e.g. audio) pass through unchanged."
    )

    @staticmethod
    def _quadrants(tensor):
        ndim = len(tensor.shape)
        if ndim == 5:
            h_dim, w_dim = 3, 4
        elif ndim == 4:
            h_dim, w_dim = 2, 3
        else:
            raise ValueError(
                f"MagikLatentSplit: expected a 4D image stream [B, C, H, W] or a 5D "
                f"video stream [B, C, T, H, W], got {ndim}D with shape "
                f"{tuple(tensor.shape)}"
            )
        h, w = tensor.shape[h_dim], tensor.shape[w_dim]
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"MagikLatentSplit: cannot split H={h}, W={w} into 4 equal quadrants "
                f"(both must be even; stream shape {tuple(tensor.shape)})"
            )
        h_step, w_step = h // 2, w // 2
        quadrants = []
        for row in range(2):
            for col in range(2):
                index = [slice(None)] * ndim
                index[h_dim] = slice(row * h_step, (row + 1) * h_step)
                index[w_dim] = slice(col * w_step, (col + 1) * w_step)
                quadrants.append(tensor[tuple(index)])
        return quadrants

    def split(self, latent):
        if "samples" not in latent:
            raise ValueError("MagikLatentSplit: input latent has no 'samples' tensor")
        samples = latent["samples"]
        if getattr(samples, "is_nested", False) and hasattr(samples, "tensors"):
            streams = list(samples.tensors)
            if not streams:
                raise ValueError("MagikLatentSplit: nested latent has no streams")
            quadrants = self._quadrants(streams[0])
            passthrough = streams[1:]
            outputs = []
            for quadrant in quadrants:
                out = dict(latent)
                out["samples"] = type(samples)([quadrant, *passthrough])
                outputs.append(out)
            return tuple(outputs)
        quadrants = self._quadrants(samples)
        outputs = []
        for quadrant in quadrants:
            out = dict(latent)
            out["samples"] = quadrant
            outputs.append(out)
        return tuple(outputs)
