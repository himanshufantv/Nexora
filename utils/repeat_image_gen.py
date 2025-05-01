# utils/replicate_image_gen.py

import replicate
from dotenv import load_dotenv
import os
import re

load_dotenv()

def generate_flux_image_with_seed(prompt: str):
    input = {
        "prompt": prompt,
        "aspect_ratio": "9:16",
        "output_format": "webp",
        "prompt_upsampling": True
    }

    try:
        prediction = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input
        )

        # For some setups (e.g. hosted version), replicate.run returns just the output URL
        # If using full prediction objects with logs, use replicate.predictions.create instead:
        # prediction = replicate.predictions.create(...)

        # Try to extract seed from logs (if available)
        if hasattr(prediction, 'logs'):
            logs = prediction.logs
            match = re.search(r"Using seed: (\d+)", logs)
            seed = int(match.group(1)) if match else None
        else:
            seed = None

        image_url = prediction[0] if isinstance(prediction, list) else prediction
        return image_url, seed

    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return "", -1
