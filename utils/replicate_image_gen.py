# utils/replicate_image_gen.py

import replicate
import os
from dotenv import load_dotenv
from utils.functional_logger import log_api_call, log_error, log_entry_exit

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
    print(f"======================= IMAGE GENERATION START =======================")
    print(f"PROMPT: {prompt}")
    print(f"LORA: {hf_lora if hf_lora else 'None'}")
    print(f"ASPECT RATIO: {aspect_ratio}")
    print(f"SEED: {seed if seed is not None else 'None (will be randomly generated)'}")
    
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
        print(f"Adding LoRA to request: {hf_lora}")
        input_data["hf_lora"] = hf_lora
        
    if seed is not None:
        print(f"Adding seed to request: {seed}")
        input_data["seed"] = int(seed)

    print(f"FULL INPUT PAYLOAD: {input_data}")
    
    try:
        # First try with the new Flux model
        try:
            print(f"Calling Replicate API - Model: flux-dev-lora")
            output = client.run(
                "lucataco/flux-dev-lora:091495765fa5ef2725a175a57b276ec30dc9d39c22d30410f2ede68a3eab66b3",
                input=input_data
            )
            print(f"Received response from flux-dev-lora: {output}")
        except Exception as e:
            print(f"Error with Flux-dev-lora model: {str(e)}")
            print(f"Trying fallback model")
            # Fallback to the standard Flux model
            fallback_input = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",        
                "prompt_strength": 0.8
            }
            if seed is not None:
                fallback_input["seed"] = int(seed)
                print(f"Adding seed to fallback request: {seed}")
            
            if hf_lora and hf_lora.strip():
                print(f"WARNING: LoRA parameter not supported in fallback model, ignoring: {hf_lora}")
                
            print(f"FALLBACK INPUT PAYLOAD: {fallback_input}")
            print(f"Calling Replicate API - Model: flux-1.1-pro (fallback)")
            output = client.run(
                "black-forest-labs/flux-1.1-pro",
                input=fallback_input
            )
            print(f"Received response from flux-1.1-pro: {output}")

        # Output is list of image URLs
        if isinstance(output, list) and len(output) > 0:
            print(f"SUCCESS: Generated image URL: {output[0]}")
            print(f"======================= IMAGE GENERATION COMPLETE =======================")
            return str(output[0]), seed
        elif isinstance(output, str):
            print(f"SUCCESS: Generated image URL: {output}")
            print(f"======================= IMAGE GENERATION COMPLETE =======================")
            return str(output), seed
        else:
            print(f"ERROR: Invalid output format from Replicate: {output}")
            print(f"======================= IMAGE GENERATION FAILED =======================")
            return "", None

    except Exception as e:
        print(f"ERROR: Image generation failed: {str(e)}")
        print(f"======================= IMAGE GENERATION FAILED =======================")
        return "", None
