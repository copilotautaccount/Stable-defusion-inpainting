"""Mask generation by comparing background and object images."""

import cv2
import numpy as np


class MaskGenerator:
    """Generate binary masks by comparing background and object images.

    Detects the object added in the 'object' image by computing pixel-level
    differences against the corresponding 'background' image.
    """

    def __init__(
        self,
        blur_kernel: int = 5,
        threshold: int = 25,
        morph_kernel: int = 5,
        min_area: int = 100,
    ):
        """Initialise the mask generator.

        Args:
            blur_kernel: Gaussian blur kernel size (must be odd).
            threshold: Pixel-difference threshold for binarisation.
            morph_kernel: Morphological-operation kernel size.
            min_area: Minimum contour area to keep (pixels).
        """
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self.morph_kernel = morph_kernel
        self.min_area = min_area

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def generate_mask(
        self, background: np.ndarray, obj: np.ndarray
    ) -> np.ndarray:
        """Return a binary mask (0/255) highlighting differences.

        Args:
            background: Background image (BGR, uint8).
            obj: Object image taken from the same viewpoint (BGR, uint8).

        Returns:
            Single-channel uint8 mask where 255 = added object.
        """
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

        # Align sizes if they differ slightly
        if bg_gray.shape != obj_gray.shape:
            obj_gray = cv2.resize(
                obj_gray, (bg_gray.shape[1], bg_gray.shape[0])
            )

        diff = cv2.absdiff(bg_gray, obj_gray)
        diff = cv2.GaussianBlur(
            diff, (self.blur_kernel, self.blur_kernel), 0
        )

        _, binary = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Remove small noise contours
        binary = self._filter_small_contours(binary)

        return binary

    def generate_mask_from_paths(
        self, background_path: str, object_path: str
    ) -> np.ndarray:
        """Convenience wrapper that reads images from file paths."""
        background = cv2.imread(background_path)
        obj = cv2.imread(object_path)
        if background is None:
            raise FileNotFoundError(
                f"Cannot read background image: {background_path}"
            )
        if obj is None:
            raise FileNotFoundError(
                f"Cannot read object image: {object_path}"
            )
        return self.generate_mask(background, obj)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_small_contours(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)
        return filtered
