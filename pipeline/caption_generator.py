"""Caption generation for masked objects."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy imports – only loaded when the BLIP model is requested.
_blip_processor = None
_blip_model = None


def _load_blip():
    """Lazy-load the BLIP captioning model."""
    global _blip_processor, _blip_model  # noqa: PLW0603
    if _blip_processor is not None:
        return

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_name = "Salesforce/blip-image-captioning-base"
        logger.info("Loading BLIP model: %s …", model_name)
        _blip_processor = BlipProcessor.from_pretrained(model_name)
        _blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        logger.info("BLIP model loaded successfully.")
    except Exception:
        logger.warning(
            "Could not load BLIP model. Falling back to rule-based captions.",
            exc_info=True,
        )


class CaptionGenerator:
    """Generate captions describing the masked object region.

    Supports two modes:

    * **blip** – uses the BLIP vision-language model (requires GPU or a
      beefy CPU and ``transformers`` + ``torch``).
    * **simple** – deterministic rule-based caption derived from mask
      geometry (no ML dependencies).
    """

    def __init__(self, mode: str = "simple"):
        """Initialise the caption generator.

        Args:
            mode: ``"blip"`` for model-based captions, ``"simple"`` for
                rule-based captions.
        """
        if mode not in ("blip", "simple"):
            raise ValueError(f"Unknown caption mode: {mode!r}")
        self.mode = mode
        if mode == "blip":
            _load_blip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_caption(
        self, image: np.ndarray, mask: np.ndarray
    ) -> str:
        """Return a text caption describing the masked object.

        Args:
            image: The *object* image (BGR, uint8).
            mask: Binary mask (single-channel, 255 = object).

        Returns:
            A short English caption string.
        """
        if self.mode == "blip" and _blip_processor is not None:
            return self._caption_blip(image, mask)
        return self._caption_simple(image, mask)

    # ------------------------------------------------------------------
    # BLIP-based captioning
    # ------------------------------------------------------------------

    @staticmethod
    def _caption_blip(image: np.ndarray, mask: np.ndarray) -> str:
        from PIL import Image

        # Crop to bounding box of mask
        x, y, w, h = cv2.boundingRect(mask)
        cropped = image[y : y + h, x : x + w]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        inputs = _blip_processor(pil_img, return_tensors="pt")
        output_ids = _blip_model.generate(**inputs, max_new_tokens=50)
        caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    # ------------------------------------------------------------------
    # Rule-based captioning
    # ------------------------------------------------------------------

    @staticmethod
    def _caption_simple(image: np.ndarray, mask: np.ndarray) -> str:
        """Derive a descriptive caption from mask geometry and colour."""
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return "an object"

        img_h, img_w = image.shape[:2]
        rel_area = cv2.countNonZero(mask) / (img_h * img_w)

        # Size descriptor
        if rel_area > 0.25:
            size = "a large"
        elif rel_area > 0.05:
            size = "a medium-sized"
        else:
            size = "a small"

        # Position descriptor
        cx, cy = x + w // 2, y + h // 2
        vert = "top" if cy < img_h / 3 else ("bottom" if cy > 2 * img_h / 3 else "center")
        horiz = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
        if vert == "center" and horiz == "center":
            position = "in the center"
        elif vert == "center":
            position = f"on the {horiz}"
        elif horiz == "center":
            position = f"at the {vert}"
        else:
            position = f"at the {vert}-{horiz}"

        # Dominant colour of the masked region
        colour = _dominant_colour(image, mask)

        return f"{size} {colour} object {position} of the image"


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _dominant_colour(image: np.ndarray, mask: np.ndarray) -> str:
    """Return a human-readable name for the dominant colour inside the mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    h, s, v = mean_hsv

    if s < 40:
        if v < 60:
            return "dark"
        if v > 200:
            return "white"
        return "gray"

    # Map hue to colour name
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 80:
        return "green"
    if h < 130:
        return "blue"
    if h < 170:
        return "purple"
    return "coloured"
