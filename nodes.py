import importlib
import json
import os
from pathlib import Path
import sys
import threading

os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")

import cv2
import kornia
import numpy as np
import onnxruntime
import torch

import comfy.model_management
import comfy.utils

# The provider-aware detector below adapts the corresponding LipForcing
# FaceDetector selection logic. LipForcing is Apache-2.0; keep its checkout
# and license alongside this node when distributing the integration.


PROCESSOR_LOCK = threading.RLock()
PROCESSORS = {}

SUPPORTED_ONNX_PROVIDERS = (
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
)

# This is the landmark order used by LipForcing's pinned FaceDetector.  It is
# kept here only for the provider-aware adapter below; the actual alignment and
# mask geometry still come from the pinned LipForcing ImageProcessor.
LMK_ADAPT_ORIGIN_ORDER = [
    1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26,
    17, 43, 48, 49, 51, 50, 102, 103, 104, 105, 101, 73, 74, 86,
]


def _resolve_path(value, *, root=None):
    path = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def _validate_install(lipforcing_root, fixed_mask_path):
    root = _resolve_path(lipforcing_root)
    mask_path = _resolve_path(fixed_mask_path, root=root)

    alignment_files = tuple(
        root / relative
        for relative in (
            "OmniAvatar/utils/latentsync/image_processor.py",
            "OmniAvatar/utils/latentsync/face_detector.py",
            "OmniAvatar/utils/latentsync/affine_transform.py",
        )
    )
    missing_sources = [path for path in alignment_files if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "LipForcing alignment sources are incomplete: "
            + ", ".join(str(path) for path in missing_sources)
            + ". "
            "Point lipforcing_root at the pinned LipForcing checkout."
        )
    if not mask_path.is_file():
        raise FileNotFoundError(f"LipForcing fixed mask not found: {mask_path}")

    insightface_root = root / "checkpoints/auxiliary"
    buffalo_dir = insightface_root / "models/buffalo_l"
    required_onnx = ("det_10g.onnx", "2d106det.onnx")
    onnx_files = tuple(buffalo_dir / name for name in required_onnx)
    missing_onnx = [path.name for path in onnx_files if not path.is_file()]
    if missing_onnx:
        raise FileNotFoundError(
            f"InsightFace buffalo_l assets are incomplete under {buffalo_dir}; "
            f"missing {', '.join(missing_onnx)}. Download the assets before starting ComfyUI."
        )

    return root, mask_path, insightface_root, alignment_files + onnx_files


def _import_lipforcing(root):
    module_name = "OmniAvatar.utils.latentsync.image_processor"
    existing = sys.modules.get(module_name)
    if existing is not None:
        module_file = Path(existing.__file__).resolve()
        if root not in module_file.parents:
            raise RuntimeError(
                f"{module_name} is already imported from {module_file}, not {root}. "
                "Restart ComfyUI after correcting lipforcing_root."
            )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    if root not in module_file.parents:
        raise RuntimeError(f"Imported LipForcing from unexpected path: {module_file}")
    return module, module.ImageProcessor, module.load_fixed_mask


def _processor_key(root, mask_path, fingerprint_files, device, onnx_provider):
    file_fingerprints = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (*fingerprint_files, mask_path)
    )
    return (
        str(root),
        file_fingerprints,
        device,
        onnx_provider,
    )


def _cuda_context_id(device):
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(
            f"CUDAExecutionProvider requires a CUDA torch device, got {device!r}."
        )
    return 0 if parsed.index is None else int(parsed.index)


class _ProviderAwareFaceDetector:
    """LipForcing FaceDetector with an explicit ONNX Runtime provider.

    The pinned LipForcing checkout hard-codes CUDAExecutionProvider.  The
    preprocessing notebook, however, also has a CPU-only ONNX Runtime path.
    This adapter preserves LipForcing's face-selection/landmark behavior while
    making that provider choice explicit.  PyTorch alignment still runs on the
    ComfyUI CUDA device.
    """

    def __init__(self, device="cuda", insightface_root="checkpoints/auxiliary", *, provider):
        from insightface.app import FaceAnalysis

        self.provider = provider
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root=insightface_root,
            providers=[provider],
        )
        context_id = -1 if provider == "CPUExecutionProvider" else _cuda_context_id(device)
        self.app.prepare(ctx_id=context_id, det_size=(512, 512))

    def __call__(self, frame, threshold=0.5):
        f_height, f_width, _ = frame.shape
        faces = self.app.get(frame)
        selected = None
        max_size = 0

        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            if face_width < 50 or face_height < 80:
                continue
            if face_width / face_height > 1.5 or face_width / face_height < 0.2:
                continue
            if face.det_score < threshold:
                continue
            size = face_width * face_height
            if size > max_size:
                max_size = size
                selected = face

        if selected is None:
            return None, None

        landmarks = np.round(selected.landmark_2d_106).astype(np.int_)
        half_face = np.mean([landmarks[74], landmarks[73]], axis=0)
        subset = landmarks[LMK_ADAPT_ORIGIN_ORDER]
        half_face_distance = np.max(subset[:, 1]) - half_face[1]
        upper_bound = half_face[1] - half_face_distance

        x1 = int(np.min(subset[:, 0]))
        y1 = int(upper_bound)
        x2 = int(np.max(subset[:, 0]))
        y2 = int(np.max(subset[:, 1]))
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = selected.bbox.astype(np.int_).tolist()

        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)
        return (
            max(0, x1),
            max(0, y1),
            min(f_width, x2),
            min(f_height, y2),
        ), landmarks


def _get_processor(
    root,
    mask_path,
    insightface_root,
    fingerprint_files,
    device,
    onnx_provider,
):
    key = _processor_key(
        root, mask_path, fingerprint_files, device, onnx_provider
    )
    processor = PROCESSORS.get(key)
    if processor is not None:
        return processor

    image_processor_module, ImageProcessor, load_fixed_mask = _import_lipforcing(root)
    fixed_mask = load_fixed_mask(512, str(mask_path))
    # The pinned upstream constructor hard-codes CUDAExecutionProvider and
    # creates its detector before returning.  Construct it in its documented
    # CPU/no-detector mode, then attach the explicit detector and the same
    # upstream AlignRestore implementation on the ComfyUI CUDA device.  This
    # avoids mutating the imported module and preserves the upstream geometry.
    processor = ImageProcessor(
        resolution=512,
        device="cpu",
        mask_image=fixed_mask,
        insightface_root=str(insightface_root),
    )
    processor.restorer = image_processor_module.AlignRestore(
        resolution=512,
        device=device,
    )
    processor.face_detector = _ProviderAwareFaceDetector(
        device=device,
        insightface_root=str(insightface_root),
        provider=onnx_provider,
    )
    PROCESSORS.clear()
    PROCESSORS[key] = processor
    return processor


def _validate_images(images):
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(
            f"Expected ComfyUI IMAGE layout [frames, height, width, channels>=3], got {tuple(images.shape)}"
        )
    if images.shape[0] < 1 or images.shape[1] < 2 or images.shape[2] < 2:
        raise ValueError(f"Image batch is empty or has invalid geometry: {tuple(images.shape)}")


def _frame_to_rgb_u8(frame):
    return (
        frame[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .numpy()
    )


def _aligned_edit_mask(keep_mask):
    mask = 1.0 - keep_mask.detach().to(device="cpu", dtype=torch.float32).numpy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"LipForcing keep-mask has unsupported shape: {keep_mask.shape}")
    return np.clip(mask, 0.0, 1.0)


def _full_frame_mask(processor, rgb, height, width):
    aligned_face, _, affine = processor.affine_transform(rgb)
    _, _, keep_mask = processor.preprocess_fixed_mask_image(
        aligned_face, affine_transform=False
    )
    edit_aligned = _aligned_edit_mask(keep_mask[0])

    native_width, native_height = processor.restorer.face_size
    edit_native = cv2.resize(
        edit_aligned,
        (native_width, native_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    edit_native = np.clip(edit_native, 0.0, 1.0).astype(np.float32, copy=False)

    restorer_device = processor.restorer.device
    inverse_affine = kornia.geometry.transform.invert_affine_transform(
        affine.to(device=restorer_device, dtype=torch.float32)
    )
    native_tensor = torch.from_numpy(edit_native)[None, None].to(
        device=restorer_device, dtype=torch.float32
    )
    return kornia.geometry.transform.warp_affine(
        native_tensor,
        inverse_affine,
        (height, width),
        mode="bilinear",
        padding_mode="zeros",
    )[0, 0].clamp(0.0, 1.0).to(device="cpu", dtype=torch.float32)


class LipForcingMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "onnx_provider": (
                    list(SUPPORTED_ONNX_PROVIDERS),
                    {"default": "CPUExecutionProvider"},
                ),
                "lipforcing_root": (
                    "STRING",
                    {"default": "/content/LipForcing", "multiline": False},
                ),
                "fixed_mask_path": (
                    "STRING",
                    {
                        "default": "weights/mask(1).png",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("mask_images", "mask", "report")
    FUNCTION = "create_mask"
    CATEGORY = "MiniMax H3/Lip Sync"
    DESCRIPTION = (
        "Runs LipForcing face alignment and fixed-mask preprocessing on a ComfyUI "
        "IMAGE batch. The returned mask uses H3's convention: 1 = regenerate."
    )

    def create_mask(
        self,
        images,
        onnx_provider,
        lipforcing_root,
        fixed_mask_path,
    ):
        _validate_images(images)
        root, mask_path, insightface_root, fingerprint_files = _validate_install(
            lipforcing_root, fixed_mask_path
        )
        if onnx_provider not in SUPPORTED_ONNX_PROVIDERS:
            raise ValueError(
                f"Unsupported ONNX provider {onnx_provider!r}; choose one of "
                f"{', '.join(SUPPORTED_ONNX_PROVIDERS)}."
            )
        onnx_providers = tuple(onnxruntime.get_available_providers())
        if onnx_provider not in onnx_providers:
            raise RuntimeError(
                f"Requested {onnx_provider}, but this ONNX Runtime build exposes "
                f"{', '.join(onnx_providers) or '(none)'}. Install the matching "
                "ONNX Runtime build or choose an available provider."
            )

        torch_device = comfy.model_management.get_torch_device()
        if torch_device.type != "cuda":
            raise RuntimeError(
                "LipForcing ImageProcessor disables face detection when its PyTorch device is CPU. "
                "Run this H3 workflow on a CUDA ComfyUI device. ONNX Runtime may still use CPU."
            )
        device = str(torch_device)

        frame_count, height, width, _ = map(int, images.shape)
        masks = []
        progress = comfy.utils.ProgressBar(frame_count)

        with PROCESSOR_LOCK:
            processor = _get_processor(
                root,
                mask_path,
                insightface_root,
                fingerprint_files,
                device,
                onnx_provider,
            )
            processor.restorer.p_bias = None

            for index, frame in enumerate(images):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if not torch.isfinite(frame).all().item():
                    raise ValueError(f"Source image contains NaN or infinite values at frame {index}.")
                try:
                    mask = _full_frame_mask(
                        processor,
                        _frame_to_rgb_u8(frame),
                        height,
                        width,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"LipForcing face alignment failed at source frame {index}: {exc}"
                    ) from exc
                if not torch.isfinite(mask).all().item():
                    raise RuntimeError(f"LipForcing produced a non-finite mask at frame {index}.")
                if torch.count_nonzero(mask > 0.05).item() == 0:
                    raise RuntimeError(f"LipForcing produced an empty mask at frame {index}.")
                masks.append(mask)
                progress.update(1)

        mask_batch = torch.stack(masks, dim=0)
        mask_images = (
            mask_batch.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        )
        coverage = (mask_batch > 0.05).to(dtype=torch.float32).mean(dim=(1, 2))
        report = json.dumps(
            {
                "frames": frame_count,
                "width": width,
                "height": height,
                "mask_convention": "1=regenerate, 0=preserve",
                "processing_device": device,
                "onnx_provider_requested": onnx_provider,
                "onnx_available_providers": list(onnx_providers),
                "coverage_over_0_05": {
                    "minimum": round(float(coverage.min().item()), 6),
                    "mean": round(float(coverage.mean().item()), 6),
                    "maximum": round(float(coverage.max().item()), 6),
                },
                "lipforcing_root": str(root),
                "fixed_mask_path": str(mask_path),
            },
            sort_keys=True,
        )
        return mask_images, mask_batch, report


__all__ = ["LipForcingMask"]
