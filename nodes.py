import torch


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


class MagikLatentMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_left": ("LATENT",),
                "top_right": ("LATENT",),
                "bottom_left": ("LATENT",),
                "bottom_right": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "merge"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Recombines 4 quadrant latents (top_left, top_right, bottom_left, "
        "bottom_right) into a single latent. Inverse of Magik Latent Split. For "
        "nested latents (e.g. MiniMax H3 AV) the first streams are merged and the "
        "remaining streams (e.g. audio) are taken from the top_left input."
    )

    @staticmethod
    def _merge_streams(quadrants):
        ndim = len(quadrants[0].shape)
        if ndim == 5:
            h_dim, w_dim = 3, 4
        elif ndim == 4:
            h_dim, w_dim = 2, 3
        else:
            raise ValueError(
                f"MagikLatentMerge: expected 4D image quadrants [B, C, H, W] or 5D "
                f"video quadrants [B, C, T, H, W], got {ndim}D with shape "
                f"{tuple(quadrants[0].shape)}"
            )
        base = quadrants[0].shape
        for q in quadrants[1:]:
            if len(q.shape) != ndim:
                raise ValueError(
                    f"MagikLatentMerge: quadrant rank mismatch, got {ndim}D and "
                    f"{len(q.shape)}D"
                )
            for i in range(ndim):
                if i not in (h_dim, w_dim) and q.shape[i] != base[i]:
                    raise ValueError(
                        f"MagikLatentMerge: quadrants must match on batch/channel/"
                        f"frame axes, got {tuple(base)} vs {tuple(q.shape)}"
                    )
        tl, tr, bl, br = quadrants
        if tl.shape[h_dim] != tr.shape[h_dim] or bl.shape[h_dim] != br.shape[h_dim]:
            raise ValueError(
                f"MagikLatentMerge: top-row and bottom-row quadrant heights must "
                f"match pairwise, got {tuple(tl.shape)}, {tuple(tr.shape)}, "
                f"{tuple(bl.shape)}, {tuple(br.shape)}"
            )
        if tl.shape[w_dim] != bl.shape[w_dim] or tr.shape[w_dim] != br.shape[w_dim]:
            raise ValueError(
                f"MagikLatentMerge: left-column and right-column quadrant widths "
                f"must match pairwise, got {tuple(tl.shape)}, {tuple(tr.shape)}, "
                f"{tuple(bl.shape)}, {tuple(br.shape)}"
            )
        top = torch.cat([tl, tr], dim=w_dim)
        bottom = torch.cat([bl, br], dim=w_dim)
        return torch.cat([top, bottom], dim=h_dim)

    def merge(self, top_left, top_right, bottom_left, bottom_right):
        names = ("top_left", "top_right", "bottom_left", "bottom_right")
        latents = (top_left, top_right, bottom_left, bottom_right)
        for name, lat in zip(names, latents):
            if "samples" not in lat:
                raise ValueError(f"MagikLatentMerge: input '{name}' has no 'samples' tensor")
        samples_list = [lat["samples"] for lat in latents]
        first = samples_list[0]
        if getattr(first, "is_nested", False) and hasattr(first, "tensors"):
            if not first.tensors:
                raise ValueError("MagikLatentMerge: nested input 'top_left' has no streams")
            n_streams = len(first.tensors)
            for name, s in zip(names[1:], samples_list[1:]):
                if (
                    not getattr(s, "is_nested", False)
                    or not hasattr(s, "tensors")
                    or len(s.tensors) != n_streams
                ):
                    raise ValueError(
                        f"MagikLatentMerge: input '{name}' does not have the same "
                        f"nested structure as 'top_left' ({n_streams} streams)"
                    )
            merged = self._merge_streams([s.tensors[0] for s in samples_list])
            out = dict(top_left)
            out["samples"] = type(first)([merged, *first.tensors[1:]])
            return (out,)
        merged = self._merge_streams(samples_list)
        out = dict(top_left)
        out["samples"] = merged
        return (out,)


class MagikSeparateAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "separate"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Splits a nested AV latent (e.g. MiniMax H3: video [B, 24, T, H, W] + audio "
        "[B, 32, 2, T]) into its video and audio streams as two separate latents."
    )

    def separate(self, av_latent):
        if "samples" not in av_latent:
            raise ValueError("MagikSeparateAVLatent: input latent has no 'samples' tensor")
        samples = av_latent["samples"]
        if (
            not getattr(samples, "is_nested", False)
            or not hasattr(samples, "tensors")
            or len(samples.tensors) != 2
        ):
            raise ValueError(
                "MagikSeparateAVLatent: expects a nested AV latent with exactly 2 "
                "streams (video, audio), got is_nested="
                f"{getattr(samples, 'is_nested', False)}"
            )
        video = dict(av_latent)
        video["samples"] = samples.tensors[0]
        audio = dict(av_latent)
        audio["samples"] = samples.tensors[1]
        return (video, audio)


class MagikConcatAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("LATENT",),
                "audio": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "concat"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Packs a video latent and an audio latent into a nested AV latent "
        "(inverse of Magik Separate AV Latent)."
    )

    def concat(self, video, audio):
        for name, lat in (("video", video), ("audio", audio)):
            if "samples" not in lat:
                raise ValueError(f"MagikConcatAVLatent: input '{name}' has no 'samples' tensor")
        from comfy.nested_tensor import NestedTensor

        out = dict(video)
        out["samples"] = NestedTensor((video["samples"], audio["samples"]))
        return (out,)


class MagikLatentSplitAdvanced:
    RETURN_NAMES = (
        "top_left", "tl_video",
        "top_right", "tr_video",
        "bottom_left", "bl_video",
        "bottom_right", "br_video",
        "audio",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",) * 9
    FUNCTION = "split"
    CATEGORY = "latent/magik"
    DESCRIPTION = (
        "Advanced Magik Latent Split: for each of the 4 spatial quadrants it outputs "
        "the quadrant latent plus the single video stream latent, and one shared "
        "audio output. For nested AV latents (e.g. MiniMax H3) the quadrant outputs "
        "stay nested and the audio output carries the full (uncut) audio stream. "
        "For plain latents the *_video outputs mirror the quadrant and the audio "
        "output passes the input through uncut."
    )

    def split(self, latent):
        if "samples" not in latent:
            raise ValueError("MagikLatentSplitAdvanced: input latent has no 'samples' tensor")
        samples = latent["samples"]
        if getattr(samples, "is_nested", False) and hasattr(samples, "tensors") and samples.tensors:
            streams = list(samples.tensors)
            audio_stream = streams[1] if len(streams) > 1 else streams[0]
            passthrough = streams[1:]
            quadrants = MagikLatentSplit._quadrants(streams[0])
            outputs = []
            for quadrant in quadrants:
                quad_out = dict(latent)
                quad_out["samples"] = type(samples)([quadrant, *passthrough])
                video_out = dict(latent)
                video_out["samples"] = quadrant
                outputs.extend((quad_out, video_out))
            audio_out = dict(latent)
            audio_out["samples"] = audio_stream
            outputs.append(audio_out)
            return tuple(outputs)
        quadrants = MagikLatentSplit._quadrants(samples)
        outputs = []
        for quadrant in quadrants:
            quad_out = dict(latent)
            quad_out["samples"] = quadrant
            video_out = dict(latent)
            video_out["samples"] = quadrant
            outputs.extend((quad_out, video_out))
        audio_out = dict(latent)
        audio_out["samples"] = samples
        outputs.append(audio_out)
        return tuple(outputs)
