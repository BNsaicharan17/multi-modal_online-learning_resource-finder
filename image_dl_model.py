"""
Image-based topic prediction for learning resources
Compatible with current requirements (no TensorFlow/Keras)
"""

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Topic mappings (kept for future extension if needed)
# ---------------------------------------------------------------------------
LABEL_TOPIC_MAP = {
    "abacus": "Data Science",
    "calculator": "Data Science",
    "slide_rule": "Data Science",
    "book_jacket": "Python",
    "binder": "Python",
    "notebook": "Python",
    "library": "Python",
    "brain_coral": "Deep Learning",
    "maze": "Deep Learning",
    "spider_web": "Deep Learning",
    "camera": "Computer Vision",
    "microscope": "Computer Vision",
    "server": "Cloud Computing",
    "monitor": "Web Development",
    "keyboard": "Web Development",
    "mouse": "Web Development",
    "typewriter_keyboard": "NLP",
    "envelope": "NLP",
    "laptop": "Machine Learning",
}


KEYWORD_TOPIC_MAP = {
    "computer": "Machine Learning",
    "laptop": "Machine Learning",
    "notebook": "Python",
    "book": "Python",
    "library": "Python",
    "network": "Deep Learning",
    "web": "Web Development",
    "screen": "Web Development",
    "monitor": "Web Development",
    "keyboard": "NLP",
    "camera": "Computer Vision",
    "microscope": "Computer Vision",
    "server": "Cloud Computing",
    "calculator": "Data Science",
}


# ---------------------------------------------------------------------------
# Topic prediction function
# ---------------------------------------------------------------------------
def predict_image(img):
    """
    Predict learning topic from an uploaded image.
    Uses simple image statistics since deep learning libraries are not used.
    """

    try:
        # Convert and resize image
        img = img.convert("RGB")
        img = img.resize((224, 224))

        # Convert to numpy array
        img_array = np.array(img)

        # Calculate simple features
        mean_val = np.mean(img_array)
        std_val = np.std(img_array)

        # Simple topic decision logic
        if mean_val > 150:
            return "Web Development"

        elif std_val > 70:
            return "Computer Vision"

        elif mean_val > 120:
            return "Machine Learning"

        elif std_val > 50:
            return "Data Science"

        else:
            return "Python"

    except Exception:
        return None
