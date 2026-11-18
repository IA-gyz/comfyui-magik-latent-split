# ComfyUI Magik Latent Split

Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that splits one
latent into **4 equal latents** spatially — a 2×2 grid of quadrants.

## Install

Clone this repository into ComfyUI's `custom_nodes` directory and restart ComfyUI:

```bash
git clone https://github.com/IA-gyz/comfyui-magik-latent-split.git \
  ComfyUI/custom_nodes/comfyui-magik-latent-split
```

## Usage

Add the **Magik Latent Split** node (category `latent/magik`), feed a `LATENT` into it,
and connect the four outputs `top_left`, `top_right`, `bottom_left`, `bottom_right`.

Example: a 256×256 latent is split into four **128×128** latents (each side is halved,
so 4 pieces tile the original). Note that 64×64 pieces from a 256×256 latent would be a
4×4 grid, i.e. 16 latents, not 4.

## Magik Latent Merge

The **Magik Latent Merge** node (category `latent/magik`) is the inverse of the
split: feed it the four quadrant outputs (`top_left`, `top_right`, `bottom_left`,
`bottom_right`) and it returns a single recombined latent. Quadrants are joined
column-wise in each row, then row-wise (`cat` along W, then along H). For nested
AV latents the video streams are merged and the audio stream is taken from the
`top_left` input (all four carry the same audio after a split). Quadrant shapes
must be consistent, otherwise the node raises a clear error.

## What each split output keeps

Each output latent keeps the full channel depth `C` and — for video latents — **all
frames** `T`. Only the spatial plane is cut: each output holds one quadrant at
`H/2 × W/2`. Placed back into a 2×2 grid, the four outputs reconstruct the original
latent exactly. `H` and `W` must both be even, otherwise the node raises an error.

## Video vs image latents

The split is spatial for both; only the tensor rank differs, and the node picks the
spatial axes from the rank:

| Latent kind | Samples shape | Split axes | Result |
|---|---|---|---|
| Image | `[B, C, H, W]` (4D) | H, W (dims 2, 3) | 4 latents of `[B, C, H/2, W/2]` |
| Video (e.g. Wan/LTXV VAE) | `[B, C, T, H, W]` (5D) | H, W (dims 3, 4) | 4 latents of `[B, C, T, H/2, W/2]` — every frame, quarter resolution |

All other keys of the input latent dict (e.g. `noise_mask`) are passed through
unchanged to each of the four outputs.

## Magik Latent Split Advanced

The **Magik Latent Split Advanced** node (category `latent/magik`) extends the split
to 9 outputs — per quadrant the quadrant latent itself plus the single video stream,
and one shared audio output:

`top_left, tl_video, top_right, tr_video, bottom_left, bl_video, bottom_right,
br_video, audio`

For a nested AV latent (MiniMax H3) the `top_left`/... outputs stay nested AV
quadrants (same as **Magik Latent Split**), `*_video` is the plain video quadrant,
and the single `audio` output is the full audio stream (audio has no spatial plane,
so it is passed uncut — no need to duplicate it per quadrant). For a plain
(non-nested) latent the `*_video` outputs mirror the quadrant and `audio` passes the
input through uncut. The four quadrant outputs feed straight into
**Magik Latent Merge**.

## Magik Separate / Concat AV Latent

For nested AV latents (MiniMax H3: `NestedTensor([video [B, 24, T, H, W], audio
[B, 32, 2, T]])`):

- **Magik Separate AV Latent** takes the AV latent and returns two plain latents,
  `video` and `audio` (the two nested streams, unwrapped).
- **Magik Concat AV Latent** takes a `video` latent and an `audio` latent and packs
  them back into a nested AV latent — the inverse of Separate.

Typical chain: Separate AV → split `video` with **Magik Latent Split** (audio stays
whole) → process quadrants → **Magik Latent Merge** → Concat AV with the original
`audio` → VAE decode once.

## Nested / AV latents (e.g. MiniMax H3)

Some models (MiniMax H3) pack several streams into one latent: `samples` is a
ComfyUI `NestedTensor` holding `[video [B, 24, T, H, W], audio [B, 32, 2, T]]`.
The node splits the **first stream** (video/image) into the 4 quadrants and passes
all other streams (e.g. audio) through **unchanged** to every output — audio has no
spatial plane, so it must stay full-length. Each output is again a nested latent of
the same structure, so downstream nodes (sampler, VAE decode) keep working.
