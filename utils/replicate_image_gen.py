# utils/replicate_image_gen.py

import replicate
import os
from dotenv import load_dotenv
from utils.functional_logger import log_flow, log_api_call, log_error, log_entry_exit

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@log_entry_exit
def generate_flux_image(prompt: str, hf_lora: str = "", aspect_ratio="9:16", seed=None) -> tuple:
    """
    Generate an image using Flux-dev-lora model with optional seed and LoRA.
    
    Returns:
        (image_url, used_seed)
    """
    log_flow(f"Generating image with prompt: {prompt[:50]}...")
    input_data = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "jpg",
        "guidance_scale": 3.5,
        "output_quality": 100,
        "prompt_strength": 0.8,
        "num_outputs": 1,
        "num_inference_steps": 28,
    }
    
    # Only add hf_lora if it's provided and not empty
    if hf_lora and hf_lora.strip():
        log_flow(f"Using LoRA: {hf_lora}")
        input_data["hf_lora"] = hf_lora
        
    if seed is not None:
        log_flow(f"Using seed: {seed}")
        input_data["seed"] = int(seed)

    try:
        # First try with the new Flux model
        try:
            log_api_call("Replicate - flux-dev-lora", {"prompt_length": len(prompt), "with_lora": bool(hf_lora)})
            output = client.run(
                "lucataco/flux-dev-lora:091495765fa5ef2725a175a57b276ec30dc9d39c22d30410f2ede68a3eab66b3",
                input=input_data
            )
        except Exception as e:
            log_error(f"Error with Flux-dev-lora model, trying fallback", e)
            # Fallback to the standard Flux model
            fallback_input = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "prompt_strength": 0.8
            }
            if seed is not None:
                fallback_input["seed"] = int(seed)
            
            log_api_call("Replicate - flux-1.1-pro (fallback)", {"prompt_length": len(prompt)})
            output = client.run(
                "black-forest-labs/flux-1.1-pro",
                input=fallback_input
            )

        # Output is list of image URLs
        if isinstance(output, list) and len(output) > 0:
            log_flow("Successfully generated image")
            return str(output[0]), seed
        elif isinstance(output, str):
            log_flow("Successfully generated image")
            return str(output), seed
        else:
            log_error("Invalid output format from Replicate")
            return "", None

    except Exception as e:
        log_error(f"Error generating image", e)
        return "", None
