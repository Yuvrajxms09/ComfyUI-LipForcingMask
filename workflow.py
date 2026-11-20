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

    The helper preserves the prototype's mask ordering exactly:

        LipForcing MASK -> MaskToImage -> temporal padding -> resize to the
        H3 canvas -> ImageToMask -> GrowMaskWithBlur

    H3's required latent length can exceed the source frame count, so the
    existing temporal padding branch remains before spatial resize and mask
    growth.  LipForcing itself still emits exactly one full-frame mask per
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
    # Preserve the tested graph's ordering.  LipForcing emits the H3-polarity
    # MASK; the existing MaskToImage node makes it compatible with the
    # temporal padding branch.  Padding happens before the existing resize to
    # H3's canvas, ImageToMask restores MASK type, and GrowMaskWithBlur then
    # runs at the same resolution and with the same settings as the tested
    # graph.
    wired[MASK_TO_IMAGE_NODE_ID]["inputs"]["mask"] = [NODE_ID, MASK_OUTPUT]
    wired[LAST_MASK_FRAME_NODE_ID]["inputs"]["image"] = [MASK_TO_IMAGE_NODE_ID, 0]
    wired[MASK_BATCH_NODE_ID]["inputs"]["image_1"] = [MASK_TO_IMAGE_NODE_ID, 0]
    wired[MASK_RESIZE_NODE_ID]["inputs"]["image"] = [MASK_BATCH_NODE_ID, 0]
    wired[MASK_CONVERT_NODE_ID]["inputs"]["image"] = [MASK_RESIZE_NODE_ID, 0]
    wired[GROWN_MASK_NODE_ID]["inputs"]["mask"] = [MASK_CONVERT_NODE_ID, 0]

    # These consumers intentionally remain on the grown-mask output.  The
    # original graph uses the same node for subject crops and compositing.
    for node_id in MASK_CROP_NODE_IDS:
        wired[node_id]["inputs"]["masks"] = [GROWN_MASK_NODE_ID, 0]
    wired[COMPOSITE_NODE_ID]["inputs"]["mask"] = [GROWN_MASK_NODE_ID, 0]
    return wired


__all__ = ["wire_prototype_workflow"]
