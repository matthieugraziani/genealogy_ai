"""
Moteur OCR multi-backend.
Préférez PaddleOCR pour les documents français anciens.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np


OCRBackend = Literal["paddle", "easyocr", "tesseract"]


class OCRReader:
    """
    Wrapper unifié autour de PaddleOCR, EasyOCR et Tesseract.
    Usage :
        reader = OCRReader(backend="paddle")
        text = reader.read("naissance.jpg")
    """

    def __init__(self, backend: OCRBackend = "paddle", lang: str = "fr"):
        self.backend = backend
        self.lang = lang
        self._engine = None

    def _load_engine(self):
        if self._engine is not None:
            return
        if self.backend == "paddle":
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(lang=self.lang, show_log=False, use_angle_cls=True)
        elif self.backend == "easyocr":
            import easyocr
            self._engine = easyocr.Reader([self.lang], gpu=False)
        elif self.backend == "tesseract":
            import pytesseract  # noqa — engine n'est pas un objet, on stocke le module
            self._engine = pytesseract
        else:
            raise ValueError(f"Backend inconnu : {self.backend}")

    def read(self, image_path: str | Path | np.ndarray) -> str:
        """Extrait le texte d'une image. Retourne une chaîne de caractères."""
        self._load_engine()
        if self.backend == "paddle":
            return self._read_paddle(image_path)
        elif self.backend == "easyocr":
            return self._read_easyocr(image_path)
        elif self.backend == "tesseract":
            return self._read_tesseract(image_path)

    def read_with_boxes(self, image_path: str | Path) -> list[dict]:
        """Retourne les blocs de texte avec leurs coordonnées bounding box."""
        self._load_engine()
        results = []
        if self.backend == "paddle":
            raw = self._engine.ocr(str(image_path))
            for line in raw[0] or []:
                box, (text, confidence) = line
                results.append({"text": text, "confidence": confidence, "box": box})
        elif self.backend == "easyocr":
            raw = self._engine.readtext(str(image_path))
            for (box, text, confidence) in raw:
                results.append({"text": text, "confidence": confidence, "box": box})
        return results

    # ---- backends privés ----

    def _read_paddle(self, path) -> str:
        result = self._engine.ocr(str(path) if not isinstance(path, np.ndarray) else path)
        lines = [line[1][0] for line in (result[0] or [])]
        return "\n".join(lines)

    def _read_easyocr(self, path) -> str:
        result = self._engine.readtext(str(path) if not isinstance(path, np.ndarray) else path)
        return "\n".join([text for (_, text, conf) in result if conf > 0.4])

    def _read_tesseract(self, path) -> str:
        from PIL import Image
        img = Image.open(str(path)) if not isinstance(path, np.ndarray) else Image.fromarray(path)
        config = f"--oem 3 --psm 6 -l {self.lang}"
        return self._engine.image_to_string(img, config=config)
