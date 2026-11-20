import importlib.util
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def prototype_graph():
    """Small standalone fixture covering every link rewired by the helper."""
    return {
        "9651": {"class_type": "VHS_LoadVideo", "inputs": {}},
        "836": {"class_type": "VHS_LoadVideo", "inputs": {}},
        "9608:9606": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["836", 0]},
        },
        "9608:9605": {
            "class_type": "ImageBatchMulti",
            "inputs": {
                "image_1": ["836", 0],
                "image_2": ["9608:9604", 0],
            },
        },
        "9608:9604": {
            "class_type": "RepeatImageBatch",
            "inputs": {"image": ["9608:9606", 0]},
        },
        "914": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["756", 0]},
        },
        "840": {
            "class_type": "ImageResizeKJv2",
            "inputs": {"image": ["9608:9605", 0]},
        },
        "842": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["840", 0]},
        },
        "756": {
            "class_type": "GrowMaskWithBlur",
            "inputs": {"mask": ["842", 0]},
        },
        "912": {
            "class_type": "MVEx_SubjectCrop",
            "inputs": {"masks": ["756", 0]},
        },
        "913": {
            "class_type": "MVEx_SubjectCrop",
            "inputs": {
                "original_images": ["914", 0],
                "masks": ["756", 0],
            },
        },
        "923": {
            "class_type": "MVEx_SubjectCrop",
            "inputs": {"masks": ["756", 0]},
        },
        "772": {
            "class_type": "ImageCompositeMasked",
            "inputs": {"mask": ["756", 0]},
        },
    }


def load_workflow_module():
    spec = importlib.util.spec_from_file_location(
        "lipforcing_workflow", PACKAGE_ROOT / "workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.module = load_workflow_module()
        self.original = prototype_graph()

    def test_preserves_tested_mask_order_and_padding(self):
        wired = self.module.wire_prototype_workflow(self.original)

        self.assertIn("836", self.original)
        self.assertNotIn("836", wired)
        self.assertEqual(wired["lipforcing_mask"]["class_type"], "LipForcingMask")
        self.assertEqual(wired["lipforcing_mask"]["inputs"]["images"], ["9651", 0])
        self.assertEqual(
            wired["lipforcing_mask"]["inputs"]["onnx_provider"],
            "CPUExecutionProvider",
        )
        self.assertEqual(wired["914"]["inputs"]["mask"], ["lipforcing_mask", 1])
        self.assertEqual(wired["9608:9606"]["inputs"]["image"], ["914", 0])
        self.assertEqual(wired["9608:9605"]["inputs"]["image_1"], ["914", 0])
        self.assertEqual(wired["840"]["inputs"]["image"], ["9608:9605", 0])
        self.assertEqual(wired["842"]["inputs"]["image"], ["840", 0])
        self.assertEqual(wired["756"]["inputs"]["mask"], ["842", 0])
        for node_id in ("912", "913", "923", "772"):
            self.assertEqual(
                wired[node_id]["inputs"]["masks" if node_id != "772" else "mask"],
                ["756", 0],
            )
        self.assertEqual(wired["913"]["inputs"]["original_images"], ["914", 0])
        self.assertEqual(wired["840"], self.original["840"])

    def test_does_not_mutate_input_graph(self):
        before = repr(self.original)
        self.module.wire_prototype_workflow(self.original)
        self.assertEqual(repr(self.original), before)

    def test_rejects_wrong_workflow(self):
        with self.assertRaises(KeyError):
            self.module.wire_prototype_workflow({})

    def test_provider_can_be_selected_explicitly(self):
        wired = self.module.wire_prototype_workflow(
            self.original,
            onnx_provider="CUDAExecutionProvider",
        )
        self.assertEqual(
            wired["lipforcing_mask"]["inputs"]["onnx_provider"],
            "CUDAExecutionProvider",
        )

    def test_deleted_loader_has_no_remaining_links(self):
        wired = self.module.wire_prototype_workflow(self.original)
        dangling = []
        for node_id, node in wired.items():
            for input_name, value in node.get("inputs", {}).items():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and value[0] == "836"
                ):
                    dangling.append((node_id, input_name))
        self.assertEqual(dangling, [])


if __name__ == "__main__":
    unittest.main()
