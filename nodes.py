import os
import requests
import time
import tempfile
from typing import Tuple, Optional, Dict, Any
from urllib.parse import urlparse
from PIL import Image
import torch
import numpy as np

import folder_paths
from comfy.utils import ProgressBar


class UpsamplerSmartUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False}),
                "input_image_type": (["realism", "anime", "universal"],),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "global_creativity": ("INT", {"default": 5, "min": 0, "max": 10}),
                "detail": ("INT", {"default": 5, "min": 0, "max": 10}),
            },
            "optional": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "should_enhance_faces": ("BOOLEAN", {"default": False}),
                "should_preserve_blur": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"

    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, global_creativity: int, detail: int,
                description: str = "", should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        # Convert ComfyUI image tensor to PIL Image
        pil_image = self._tensor_to_pil(image)
        
        # Upload image to temporary hosting service or save locally and create accessible URL
        image_url = self._upload_image_temp(pil_image)
        
        # Prepare API request
        payload = {
            "input": {
                "imageUrl": image_url,
                "inputImageType": input_image_type,
                "upscaleFactor": upscale_factor,
                "globalCreativity": global_creativity,
                "detail": detail,
                "description": description,
                "shouldEnhanceFaces": should_enhance_faces,
                "shouldPreserveBlur": should_preserve_blur
            }
        }
        
        # Submit upscaling job
        job_id = self._submit_upscale_job("smart-upscale", payload, api_key)
        
        # Poll for completion
        result_image = self._wait_for_completion(job_id, api_key)
        
        # Convert back to ComfyUI tensor format
        return (self._pil_to_tensor(result_image),)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # Convert from ComfyUI format [B, H, W, C] to PIL
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy and scale to 0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(numpy_image)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        # Convert PIL to ComfyUI tensor format [1, H, W, C]
        numpy_image = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _upload_image_temp(self, image: Image.Image) -> str:
        import base64
        import io
        
        # Convert image to base64 for temporary hosting services
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Use a temporary image hosting service like ImgBB, Imgur, or implement your own
        # For this example, we'll use a placeholder that explains what needs to be done
        
        # You need to implement one of these options:
        # 1. Upload to ImgBB API
        # 2. Upload to your own server/CDN
        # 3. Use imgur API
        # 4. Use any other public image hosting service
        
        # Example implementation for ImgBB (requires API key):
        # imgbb_api_key = "your_imgbb_api_key"
        # image_data = base64.b64encode(buffer.getvalue()).decode()
        # response = requests.post(
        #     "https://api.imgbb.com/1/upload",
        #     data={
        #         "key": imgbb_api_key,
        #         "image": image_data
        #     }
        # )
        # if response.status_code == 200:
        #     return response.json()["data"]["url"]
        
        raise Exception(
            "Image hosting not implemented. Please implement _upload_image_temp() method "
            "to upload images to a publicly accessible URL. You can use services like ImgBB, "
            "Imgur, or your own hosting solution."
        )

    def _submit_upscale_job(self, endpoint: str, payload: Dict[str, Any], api_key: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"https://upsampler.com/api/v1/{endpoint}",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        if result.get("status") == "FAILED":
            raise Exception(f"Job failed: {result.get('error')}")
        
        return result["id"]

    def _wait_for_completion(self, job_id: str, api_key: str) -> Image.Image:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        pbar = ProgressBar(100)
        progress = 0
        
        while True:
            response = requests.get(
                f"https://upsampler.com/api/v1/status/{job_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Status check failed: {response.status_code}")
            
            result = response.json()
            status = result.get("status")
            
            if status == "SUCCESS":
                # Download the result image
                image_url = result["imageUrl"]  # Use full quality PNG
                image_response = requests.get(image_url)
                
                if image_response.status_code != 200:
                    raise Exception(f"Failed to download result image: {image_response.status_code}")
                
                pbar.update(100)
                
                # Load image from response content
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.write(image_response.content)
                temp_file.close()
                
                result_image = Image.open(temp_file.name)
                os.unlink(temp_file.name)  # Clean up temp file
                
                return result_image
                
            elif status == "FAILED":
                raise Exception(f"Upscaling failed: {result.get('error')}")
            
            elif status in ["IN_PROGRESS", "IN_QUEUE"]:
                # Update progress bar
                progress = min(progress + 5, 95)
                pbar.update(progress)
                time.sleep(5)  # Wait 5 seconds before next check
            
            else:
                raise Exception(f"Unknown status: {status}")


class UpsamplerDynamicUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False}),
                "input_image_type": (["realism", "anime", "universal"],),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "global_creativity": ("INT", {"default": 5, "min": 0, "max": 10}),
                "resemblance": ("INT", {"default": 8, "min": 0, "max": 10}),
                "detail": ("INT", {"default": 5, "min": 0, "max": 10}),
            },
            "optional": {
                "description": ("STRING", {"multiline": True, "default": ""}),
                "should_enhance_faces": ("BOOLEAN", {"default": False}),
                "should_preserve_hands": ("BOOLEAN", {"default": False}),
                "should_preserve_blur": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"

    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, global_creativity: int, resemblance: int, 
                detail: int, description: str = "", should_enhance_faces: bool = False, 
                should_preserve_hands: bool = False, should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        # Reuse the same methods from SmartUpscale
        upscaler = UpsamplerSmartUpscale()
        pil_image = upscaler._tensor_to_pil(image)
        image_url = upscaler._upload_image_temp(pil_image)
        
        payload = {
            "input": {
                "imageUrl": image_url,
                "inputImageType": input_image_type,
                "upscaleFactor": upscale_factor,
                "globalCreativity": global_creativity,
                "resemblance": resemblance,
                "detail": detail,
                "description": description,
                "shouldEnhanceFaces": should_enhance_faces,
                "shouldPreserveHands": should_preserve_hands,
                "shouldPreserveBlur": should_preserve_blur
            }
        }
        
        job_id = upscaler._submit_upscale_job("dynamic-upscale", payload, api_key)
        result_image = upscaler._wait_for_completion(job_id, api_key)
        
        return (upscaler._pil_to_tensor(result_image),)


class UpsamplerPreciseUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False}),
                "input_image_type": (["realism", "anime", "universal"],),
                "upscale_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.1}),
            },
            "optional": {
                "should_enhance_faces": ("BOOLEAN", {"default": False}),
                "should_preserve_blur": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"

    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        # Reuse the same methods from SmartUpscale
        upscaler = UpsamplerSmartUpscale()
        pil_image = upscaler._tensor_to_pil(image)
        image_url = upscaler._upload_image_temp(pil_image)
        
        payload = {
            "input": {
                "imageUrl": image_url,
                "inputImageType": input_image_type,
                "upscaleFactor": upscale_factor,
                "shouldEnhanceFaces": should_enhance_faces,
                "shouldPreserveBlur": should_preserve_blur
            }
        }
        
        job_id = upscaler._submit_upscale_job("precise-upscale", payload, api_key)
        result_image = upscaler._wait_for_completion(job_id, api_key)
        
        return (upscaler._pil_to_tensor(result_image),)