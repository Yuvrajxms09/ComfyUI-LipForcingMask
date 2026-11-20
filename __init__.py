from .nodes import LipForcingH3TemporalPad, LipForcingMask


NODE_CLASS_MAPPINGS = {
    "LipForcingH3TemporalPad": LipForcingH3TemporalPad,
    "LipForcingMask": LipForcingMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipForcingH3TemporalPad": "LipForcing H3 Temporal Pad",
    "LipForcingMask": "LipForcing Mask (Full Frame)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
