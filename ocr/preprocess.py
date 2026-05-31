"""
Prétraitement d'images avant OCR.
Gère les documents anciens, manuscrits et numérisations basse qualité.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Charge une image depuis le disque (supporte TIFF, JPG, PNG, PDF converti)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Conversion en niveaux de gris."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Débruitage adaptatif (efficace sur vieux papier)."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarisation adaptative (meilleure que le seuil global sur les encres inégales)."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correction de l'inclinaison (deskew) par transformée de Hough."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Amélioration du contraste par CLAHE (utile pour les encres pâles)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def upscale_if_needed(img: np.ndarray, min_width: int = 1200) -> np.ndarray:
    """Agrandit l'image si elle est trop petite pour l'OCR."""
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def preprocess(path: str | Path) -> np.ndarray:
    """Pipeline complet de prétraitement. Retourne une image prête pour l'OCR."""
    img = load_image(path)
    img = upscale_if_needed(img)
    gray = to_grayscale(img)
    gray = enhance_contrast(gray)
    gray = denoise(gray)
    gray = deskew(gray)
    binary = binarize(gray)
    return binary


def save_preprocessed(path: str | Path, out_dir: str | Path = "data/extracted") -> Path:
    """Prétraite et sauvegarde le résultat, retourne le chemin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = preprocess(path)
    out_path = out_dir / (Path(path).stem + "_processed.png")
    cv2.imwrite(str(out_path), processed)
    return out_path
