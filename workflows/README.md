# Complete H3 FL2VA workflow

`minimax_h3_fl2va_lipforcing_api.json` is the complete API-format graph for
the full-frame LipForcing mask path and supplied-audio H3 FL2VA path.

The relevant mask branch is:

```text
VHS_LoadVideo (source)
  -> LipForcingMask
  -> MaskToImage (raw mask adapter)
  -> LipForcingH3TemporalPad
  -> ImageResizeKJv2
  -> ImageToMask
  -> GrowMaskWithBlur
```

`LipForcingH3TemporalPad` appends the source's final mask frame only when the
H3 target length is longer than the source. When the difference is zero, it
returns the source batch unchanged. The grown mask then feeds the existing H3
crop, latent-mask, and composite path. The supplied speech remains connected
through `MiniMaxH3AddGuide` and `MiniMaxH3ExactAudioLock`.

## Before queuing

Install this repository as a ComfyUI custom node and restart ComfyUI. The
workflow also requires the H3, VideoHelperSuite, KJNodes, MaskVidExperiments,
FearnWorks mask, and supplied-audio node packs used by the reference graph.
Install the pinned LipForcing checkout and its InsightFace assets as described
in the repository root README.

Update these input values in the API JSON for the current run:

| Node | Input | Value |
|---|---|---|
| `9651` | `video` | Silent source video |
| `732` | `image` | Reference frame 0 |
| `r2v_speech_audio` | `audio` | Supplied speech/audio file |
| `lipforcing_mask` | `lipforcing_root` | LipForcing checkout path on the ComfyUI host |
| `lipforcing_mask` | `onnx_provider` | `CPUExecutionProvider` or an installed CUDA provider |

The checked-in filenames are intentionally generic placeholders. The graph is
API-format and can be submitted to ComfyUI's `/prompt` endpoint or imported
through a compatible API-workflow loader.
