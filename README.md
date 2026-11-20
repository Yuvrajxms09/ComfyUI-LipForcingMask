# ComfyUI LipForcing Mask

This node moves the proven LipForcing preprocessing stage from the two prototype notebooks into ComfyUI. It accepts the decoded source-video frame batch and returns one full-frame H3 edit mask per source frame.

It does **not** run LipForcing generation, decode audio, change H3 conditioning, alter timing, or mux output audio. H3 Exact Audio Lock continues to receive the original supplied speech, and the final workflow continues to mux that audio directly.

## Runtime contract

- Input `images`: ComfyUI `IMAGE`, `[frames, height, width, channels]`, RGB in `[0, 1]`.
- Output `0 / mask_images`: three-channel `IMAGE`, `[frames, height, width, 3]`, retained for visualization or debugging.
- Output `1 / mask`: ComfyUI `MASK`, `[frames, height, width]`, in H3 polarity (white means editable).
- Output `2 / report`: JSON `STRING` with dimensions, providers and coverage statistics.
- Mask semantics: `1 = regenerate`, `0 = preserve`.
- A processor instance is cached by the LipForcing source/template fingerprint, PyTorch device and ONNX provider, but `restorer.p_bias` is reset once for every input batch so independent clips never share temporal alignment state.
- Processing is serialized because the LipForcing aligner owns mutable temporal state.
- The node exposes an explicit `onnx_provider` choice. The default is `CPUExecutionProvider`, matching the working notebook's face-detection setup; `CUDAExecutionProvider` is available when the ComfyUI environment has the matching ONNX Runtime build. PyTorch alignment still runs on ComfyUI's CUDA device.

The implementation preserves the notebook geometry exactly:

1. `ImageProcessor.affine_transform` aligns each RGB frame.
2. `preprocess_fixed_mask_image(..., affine_transform=False)` creates LipForcing's aligned keep-mask.
3. The keep-mask is inverted to H3's edit-mask convention.
4. The 512×512 mask is resized back to the aligner's native face crop with Lanczos4.
5. LipForcing's affine transform is inverted and the mask is bilinearly warped into the original full-frame dimensions with zero padding.

## Installation

Place this directory under `ComfyUI/custom_nodes/` and restart ComfyUI.

Use the same pinned LipForcing checkout validated by the prototype:

```bash
git clone https://github.com/cvlab-kaist/LipForcing.git /content/LipForcing
git -C /content/LipForcing checkout fc864771eb347ca3ccaaef9c0b583ff6ccc9f184
```

Install this node's requirements in the ComfyUI environment. The default `onnxruntime` dependency runs InsightFace on CPU, matching the working notebook environment. This is a supported path; PyTorch alignment and inverse warping use ComfyUI's CUDA device.

The fixed mask must exist at `/content/LipForcing/weights/mask(1).png` (or set another path in the node). InsightFace `buffalo_l` assets must already exist under `/content/LipForcing/checkpoints/auxiliary/models/buffalo_l/`. The node refuses incomplete local assets instead of initiating a hidden download. It also refuses a provider that is not exposed by the installed ONNX Runtime build; it does not silently fall back to a different provider.

## Exact prototype wiring

The API graph integration is intentionally narrow:

| Current node | Change |
|---|---|
| `9651` — source `VHS_LoadVideo` | Keep unchanged; connect image output 0 to `LipForcingMask.images` |
| `836` — mask `VHS_LoadVideo` | Remove |
| `914` — `MaskToImage` | Change `mask` to `["lipforcing_mask", 1]`, making the node output compatible with the existing image-based padding branch |
| `9608:9606 -> 9608:9605` | Use `914` as the source image, preserving the tested last-frame padding to H3's required temporal length |
| `840 -> 842 -> 756` | Keep the tested order: resize the padded mask to the H3 canvas, convert back to `MASK`, then grow/blur at the H3 resolution |
| `912`, `913`, `923`, `772` | Keep the existing grown-mask output from `756` so source, crop, latent mask and final composite use the same mask |
| H3 Exact Audio Lock and audio mux nodes | Keep unchanged |

`workflow.wire_prototype_workflow()` applies and validates those exact API-graph edits without changing the rest of the graph.

The node exposes the H3-polarity `MASK` output as output index `1`. The helper connects it to the existing `MaskToImage` node, then preserves the tested sequence `temporal padding → H3-canvas resize → ImageToMask → GrowMaskWithBlur`. The grown mask remains the input to the subject crops, latent mask, and final composite. This keeps LipForcing's one-mask-per-source-frame contract intact while satisfying H3's latent-shape and mask-resolution contracts.

## Failure behavior

The node fails with a frame index when alignment fails, a source frame is non-finite, or a generated mask is empty/non-finite. It also fails early for a missing LipForcing source tree, fixed mask, InsightFace detector assets, an unavailable requested ONNX provider, or a non-CUDA ComfyUI device. The `report` output records dimensions, the requested/available ONNX providers, processing device, and per-frame mask coverage statistics.
