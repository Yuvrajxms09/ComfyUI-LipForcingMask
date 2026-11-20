import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((PACKAGE_ROOT / "nodes.py").read_text())
        cls.node = next(
            item
            for item in cls.tree.body
            if isinstance(item, ast.ClassDef) and item.name == "LipForcingMask"
        )

    def class_assignment(self, name):
        return next(
            item.value
            for item in self.node.body
            if isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in item.targets)
        )

    def test_comfyui_node_contract(self):
        self.assertEqual(ast.literal_eval(self.class_assignment("RETURN_TYPES")), ("IMAGE", "MASK", "STRING"))
        self.assertEqual(ast.literal_eval(self.class_assignment("RETURN_NAMES")), ("mask_images", "mask", "report"))
        self.assertEqual(ast.literal_eval(self.class_assignment("FUNCTION")), "create_mask")

    def test_geometry_keeps_notebook_operations(self):
        source = (PACKAGE_ROOT / "nodes.py").read_text()
        for operation in (
            "processor.affine_transform",
            "processor.preprocess_fixed_mask_image",
            "cv2.INTER_LANCZOS4",
            "invert_affine_transform",
            'mode="bilinear"',
            'padding_mode="zeros"',
            "processor.restorer.p_bias = None",
        ):
            self.assertIn(operation, source)

    def test_provider_is_explicit_and_cache_is_provider_specific(self):
        source = (PACKAGE_ROOT / "nodes.py").read_text()
        self.assertIn('"CPUExecutionProvider"', source)
        self.assertIn('"CUDAExecutionProvider"', source)
        self.assertIn("onnx_provider_requested", source)
        self.assertIn("onnx_provider)", source)
        self.assertIn("Requested {onnx_provider}", source)

    def test_frame_failures_are_contextualized(self):
        source = (PACKAGE_ROOT / "nodes.py").read_text()
        self.assertIn("LipForcing face alignment failed at source frame {index}", source)

    def test_temporal_pad_handles_zero_without_empty_intermediate(self):
        source = (PACKAGE_ROOT / "nodes.py").read_text()
        self.assertIn("class LipForcingH3TemporalPad:", source)
        self.assertIn('"amount": (\n                    "INT"', source)
        self.assertIn('"min": 0', source)
        self.assertIn("if amount == 0:", source)
        self.assertIn("return (image,)", source)
        self.assertIn("last_frame[-1:].repeat((amount, 1, 1, 1))", source)


if __name__ == "__main__":
    unittest.main()
