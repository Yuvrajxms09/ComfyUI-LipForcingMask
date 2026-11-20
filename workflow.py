from copy import deepcopy


NODE_ID = "lipforcing_mask"
SOURCE_VIDEO_NODE_ID = "9651"
MASK_VIDEO_NODE_ID = "836"
LAST_MASK_FRAME_NODE_ID = "9608:9606"
MASK_BATCH_NODE_ID = "9608:9605"
MASK_TO_IMAGE_NODE_ID = "914"
MASK_RESIZE_NODE_ID = "840"
MASK_CONVERT_NODE_ID = "842"
GROWN_MASK_NODE_ID = "756"
MASK_CROP_NODE_IDS = ("912", "913", "923")
COMPOSITE_NODE_ID = "772"
MASK_OUTPUT = 1


def wire_prototype_workflow(
    graph,
    *,
    lipforcing_root="/content/LipForcing",
    onnx_provider="CPUExecutionProvider",
):
    """Replace the uploaded mask video with direct LipForcing masks.

    The helper keeps the prototype's post-mask temporal padding and resize
    nodes because H3's required latent length can exceed the source frame
    count.  LipForcing itself still emits exactly one full-frame mask per
    input frame.
    """
    required = {
        SOURCE_VIDEO_NODE_ID,
        MASK_VIDEO_NODE_ID,
        LAST_MASK_FRAME_NODE_ID,
        MASK_BATCH_NODE_ID,
        MASK_TO_IMAGE_NODE_ID,
        MASK_RESIZE_NODE_ID,
        MASK_CONVERT_NODE_ID,
        GROWN_MASK_NODE_ID,
        COMPOSITE_NODE_ID,
        *MASK_CROP_NODE_IDS,
    }
    missing = sorted(required.difference(graph))
    if missing:
        raise KeyError(f"Workflow is missing expected prototype nodes: {missing}")
    if NODE_ID in graph:
        raise KeyError(f"Workflow already contains node {NODE_ID!r}")

    wired = deepcopy(graph)
    del wired[MASK_VIDEO_NODE_ID]
    wired[NODE_ID] = {
        "inputs": {
            "images": [SOURCE_VIDEO_NODE_ID, 0],
            "onnx_provider": onnx_provider,
            "lipforcing_root": lipforcing_root,
            "fixed_mask_path": "weights/mask(1).png",
        },
        "class_type": "LipForcingMask",
        "_meta": {"title": "LipForcing Mask (Full Frame)"},
    }
    # Keep H3's tested temporal padding, but move it after the direct MASK
    # output.  The raw full-frame mask is grown first; only then is it
    # converted to IMAGE so the existing last-frame padding branch can extend
    # clips to H3's required latent length.
    wired[GROWN_MASK_NODE_ID]["inputs"]["mask"] = [NODE_ID, MASK_OUTPUT]
    wired[LAST_MASK_FRAME_NODE_ID]["inputs"]["image"] = [MASK_TO_IMAGE_NODE_ID, 0]
    wired[MASK_BATCH_NODE_ID]["inputs"]["image_1"] = [MASK_TO_IMAGE_NODE_ID, 0]
    wired[MASK_CROP_NODE_IDS[0]]["inputs"]["masks"] = [MASK_CONVERT_NODE_ID, 0]
    wired[MASK_CROP_NODE_IDS[1]]["inputs"]["original_images"] = [MASK_RESIZE_NODE_ID, 0]
    wired[MASK_CROP_NODE_IDS[1]]["inputs"]["masks"] = [MASK_CONVERT_NODE_ID, 0]
    wired[MASK_CROP_NODE_IDS[2]]["inputs"]["masks"] = [MASK_CONVERT_NODE_ID, 0]
    wired[COMPOSITE_NODE_ID]["inputs"]["mask"] = [MASK_CONVERT_NODE_ID, 0]
    return wired


__all__ = ["wire_prototype_workflow"]
