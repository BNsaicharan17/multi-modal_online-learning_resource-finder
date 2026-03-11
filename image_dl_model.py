"""
Optimized image-based topic prediction using MobileNetV2.
Features:
- Comprehensive keyword-to-topic mapping (50+ ImageNet labels)
- Uses top-3 predictions for better coverage
- Cached model loading for Streamlit
"""

import numpy as np

from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array

# ---------------------------------------------------------------------------
# Load model once at module level (wrap with @st.cache_resource in app.py)
# ---------------------------------------------------------------------------
model = MobileNetV2(weights="imagenet")

# ---------------------------------------------------------------------------
# Comprehensive ImageNet-label → learning-topic mapping
# ---------------------------------------------------------------------------
LABEL_TOPIC_MAP = {
    # Machine Learning / Data Science
    "abacus": "Data Science",
    "calculator": "Data Science",
    "slide_rule": "Data Science",

    # Python / Programming
    "book_jacket": "Python",
    "binder": "Python",
    "notebook": "Python",
    "library": "Python",
    "pencil_case": "Python",
    "ballpoint": "Python",
    "fountain_pen": "Python",

    # Deep Learning
    "brain_coral": "Deep Learning",
    "maze": "Deep Learning",
    "spider_web": "Deep Learning",

    # Computer Vision
    "camera": "Computer Vision",
    "Polaroid_camera": "Computer Vision",
    "reflex_camera": "Computer Vision",
    "binoculars": "Computer Vision",
    "lens_cap": "Computer Vision",
    "loupe": "Computer Vision",
    "magnifying_glass": "Computer Vision",
    "microscope": "Computer Vision",
    "projector": "Computer Vision",
    "sunglasses": "Computer Vision",
    "sunglass": "Computer Vision",

    # Cloud Computing / Infrastructure
    "server": "Cloud Computing",
    "hard_disc": "Cloud Computing",
    "switch": "Cloud Computing",
    "modem": "Cloud Computing",
    "power_drill": "Cloud Computing",

    # Web Development
    "web_site": "Web Development",
    "monitor": "Web Development",
    "screen": "Web Development",
    "desktop_computer": "Web Development",
    "keyboard": "Web Development",
    "mouse": "Web Development",
    "iPod": "Web Development",
    "cellular_telephone": "Web Development",
    "dial_telephone": "Web Development",
    "hand-held_computer": "Web Development",

    # NLP
    "typewriter_keyboard": "NLP",
    "letter_opener": "NLP",
    "envelope": "NLP",
    "comic_book": "NLP",
    "crossword_puzzle": "NLP",
    "menu": "NLP",

    # General tech → Machine Learning
    "laptop": "Machine Learning",
    "notebook_computer": "Machine Learning",
    "computer_keyboard": "Machine Learning",
    "space_bar": "Machine Learning",
    "printer": "Machine Learning",
    "joystick": "Machine Learning",
}

# Keyword substrings to check inside label names
KEYWORD_TOPIC_MAP = {
    "computer": "Machine Learning",
    "laptop": "Machine Learning",
    "notebook": "Python",
    "book": "Python",
    "library": "Python",
    "pen": "Python",
    "network": "Deep Learning",
    "web": "Web Development",
    "screen": "Web Development",
    "monitor": "Web Development",
    "keyboard": "NLP",
    "camera": "Computer Vision",
    "lens": "Computer Vision",
    "microscope": "Computer Vision",
    "server": "Cloud Computing",
    "disc": "Cloud Computing",
    "calculator": "Data Science",
    "rule": "Data Science",
}


def _map_label_to_topic(label: str) -> str | None:
    """Map an ImageNet label to a learning topic."""
    # 1. Exact match
    if label in LABEL_TOPIC_MAP:
        return LABEL_TOPIC_MAP[label]

    # 2. Keyword substring match
    label_lower = label.lower().replace("_", " ")
    for keyword, topic in KEYWORD_TOPIC_MAP.items():
        if keyword in label_lower:
            return topic

    return None


def predict_image(img):
    """Predict learning topic from an uploaded image.

    Uses top-5 MobileNetV2 predictions and maps them through the
    comprehensive label dictionary for better accuracy.

    Returns:
        str | None: The detected topic name, or None if the image
                    is not relevant to any known learning topic.
    """
    try:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=5)  # top-5 for better coverage

        # Minimum confidence threshold for a matched label to be accepted
        CONFIDENCE_THRESHOLD = 0.03  # 3%

        # Check all top-5 predictions against our mapping
        for _, label, confidence in decoded[0]:
            topic = _map_label_to_topic(label)
            if topic is not None and confidence >= CONFIDENCE_THRESHOLD:
                return topic

        # No label matched any known topic → image is NOT relevant
        return None

    except Exception as e:
        return None
