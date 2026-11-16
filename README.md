# ComfyUI Magik Latent Split

Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that splits one
latent into **4 equal latents**.

## Install

Clone this repository into ComfyUI's `custom_nodes` directory and restart ComfyUI:

```bash
git clone https://github.com/IA-gyz/comfyui-magik-latent-split.git \
  ComfyUI/custom_nodes/comfyui-magik-latent-split
```

## Usage

Add the **Magik Latent Split** node (category `latent/magik`), feed a `LATENT` into it,
and connect the four outputs `latent_1` .. `latent_4` wherever you need them.

## Video vs image latents

Yes, it makes a difference — the two latent kinds have different tensor ranks, and the
node picks the split axis from the rank:

| Latent kind | Samples shape | Split axis | Result |
|---|---|---|---|
| Video (e.g. Wan/LTXV VAE) | `[B, C, T, H, W]` (5D) | frames `T` (dim 2) | 4 latents, each with a quarter of the frames |
| Image | `[B, C, H, W]` (4D) | batch `B` (dim 0) | 4 latents, each with a quarter of the batch |

An image latent is 4D and has no frame axis, so it can only be split along the batch
dimension. A single image (batch size 1) therefore cannot be split and the node raises
a clear error. The same error is raised whenever the split dimension is not divisible
by 4 (e.g. a video with 7 frames).

All other keys of the input latent dict (e.g. `noise_mask`) are passed through
unchanged to each of the four outputs.
