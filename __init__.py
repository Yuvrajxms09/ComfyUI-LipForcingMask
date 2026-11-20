from .nodes import LipForcingMask


NODE_CLASS_MAPPINGS = {
    "LipForcingMask": LipForcingMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipForcingMask": "LipForcing Mask (Full Frame)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
