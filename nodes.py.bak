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
                "imgbb_api_key": ("STRING", {"multiline": False, "default": ""}),
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
                imgbb_api_key: str = "", description: str = "", should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        print(f"🚀 [Upsampler] Starting Smart Upscale process...")
        print(f"📏 [Upsampler] Input image shape: {image.shape}")
        print(f"🎛️ [Upsampler] Parameters - Type: {input_image_type}, Factor: {upscale_factor}, Creativity: {global_creativity}, Detail: {detail}")
        
        # Convert ComfyUI image tensor to PIL Image
        pil_image = self._tensor_to_pil(image)
        print(f"🖼️ [Upsampler] Converted to PIL image: {pil_image.size} ({pil_image.mode})")
        
        # Get ImgBB API key from parameter or environment variable
        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        print(f"🔑 [Upsampler] Image hosting: {'ImgBB (with API key)' if hosting_api_key else 'Free services (no API key)'}")
        
        # Upload image to hosting service
        print(f"📤 [Upsampler] Uploading image to hosting service...")
        image_url = self._upload_image_temp(pil_image, hosting_api_key)
        print(f"✅ [Upsampler] Image uploaded successfully: {image_url}")
        
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
        print(f"🚀 [Upsampler] Submitting job to Upsampler API...")
        job_id = self._submit_upscale_job("smart-upscale", payload, api_key)
        print(f"✅ [Upsampler] Job submitted successfully! Job ID: {job_id}")
        
        # Poll for completion
        print(f"⏳ [Upsampler] Waiting for upscaling to complete...")
        result_image = self._wait_for_completion(job_id, api_key)
        print(f"🎉 [Upsampler] Upscaling completed! Result image: {result_image.size} ({result_image.mode})")
        
        # Convert back to ComfyUI tensor format
        result_tensor = self._pil_to_tensor(result_image)
        print(f"📤 [Upsampler] Final tensor shape: {result_tensor.shape}")
        
        # Final validation
        self._validate_output(result_tensor, result_image)
        
        print(f"✨ [Upsampler] Smart Upscale process completed successfully!")
        
        return (result_tensor,)
    
    def _validate_output(self, tensor: torch.Tensor, image: Image.Image):
        """Validate the final output before returning"""
        print(f"🔍 [Final Validation] Starting output validation...")
        
        # Check tensor properties
        print(f"📊 [Validation] Tensor - Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
        print(f"📊 [Validation] Image - Size: {image.size}, Mode: {image.mode}, Format: {image.format}")
        
        # Calculate size increase
        if len(tensor.shape) == 4:
            height, width = tensor.shape[1], tensor.shape[2]
        else:
            height, width = tensor.shape[0], tensor.shape[1]
            
        print(f"📏 [Validation] Output dimensions: {width}x{height}")
        
        # Validate tensor is not empty or corrupted
        if tensor.numel() == 0:
            print(f"❌ [Validation] ERROR: Tensor is empty!")
            raise Exception("Output tensor is empty")
        
        if torch.isnan(tensor).any():
            print(f"❌ [Validation] ERROR: Tensor contains NaN values!")
            raise Exception("Output tensor contains NaN values")
            
        if torch.isinf(tensor).any():
            print(f"❌ [Validation] ERROR: Tensor contains infinite values!")
            raise Exception("Output tensor contains infinite values")
        
        print(f"✅ [Validation] Output validation passed successfully!")
        print(f"🎯 [Validation] Image ready for ComfyUI pipeline!")

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # Convert from ComfyUI format [B, H, W, C] to PIL
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy and scale to 0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(numpy_image)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        # Convert PIL to ComfyUI tensor format [1, H, W, C]
        print(f"🔄 [Conversion] Converting PIL image to tensor: {image.size} -> tensor")
        
        # Validate image
        if image.mode not in ['RGB', 'RGBA']:
            print(f"⚠️ [Conversion] Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        numpy_image = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        print(f"✅ [Conversion] Conversion successful: {tensor.shape}")
        
        # Validate tensor values
        if tensor.min() < 0 or tensor.max() > 1:
            print(f"⚠️ [Validation] Tensor values outside [0,1]: min={tensor.min():.3f}, max={tensor.max():.3f}")
        else:
            print(f"✅ [Validation] Tensor values valid: min={tensor.min():.3f}, max={tensor.max():.3f}")
            
        return tensor

    def _upload_image_temp(self, image: Image.Image, imgbb_api_key: str = None) -> str:
        import base64
        import io
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode()
        image_size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        print(f"📊 [Upload] Image size: {image_size_mb:.2f} MB")
        
        # Check size limits and provide guidance
        if image_size_mb > 200:
            print(f"⚠️ [Upload] Warning: Image is {image_size_mb:.2f} MB, exceeding all free service limits")
            print(f"💡 [Upload] Consider compressing the image significantly")
        elif image_size_mb > 32:
            print(f"⚠️ [Upload] Warning: Image is {image_size_mb:.2f} MB, exceeding ImgBB's 32MB limit")
            print(f"💡 [Upload] Will try Catbox.moe (200MB limit) if ImgBB fails")
        
        # Use ImgBB if API key provided
        if imgbb_api_key:
            return self._upload_to_imgbb(image_data, imgbb_api_key)
        
        # Try free services without API key
        try:
            return self._upload_to_free_service(image_data)
        except Exception as e:
            raise Exception(
                f"Image upload failed: {str(e)}\n\n"
                "To resolve this, you have several options:\n"
                "1. Get a free ImgBB API key from https://api.imgbb.com/ (32MB limit)\n"
                "2. Add 'imgbb_api_key' parameter to the node input\n"
                "3. Set IMGBB_API_KEY environment variable\n"
                "4. Compress your image (under 32MB for ImgBB, under 200MB for free services)\n\n"
                "Free services tried: Catbox.moe (200MB), 0x0.st, PostImages\n"
                "ImgBB with API key is most reliable for consistent uploads."
            )

    def _upload_to_imgbb(self, image_data: str, api_key: str) -> str:
        """Upload image to ImgBB service"""
        print(f"🔄 [ImgBB] Uploading to ImgBB (expires in 1 hour)...")
        
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={
                "key": api_key,
                "image": image_data,
                "expiration": 3600  # 1 hour expiration
            }
        )
        
        print(f"📡 [ImgBB] Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                url = result["data"]["url"]
                print(f"✅ [ImgBB] Upload successful: {url}")
                return url
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                print(f"❌ [ImgBB] Upload failed: {error_msg}")
                raise Exception(f"ImgBB upload failed: {error_msg}")
        else:
            print(f"❌ [ImgBB] API error: {response.status_code} - {response.text}")
            raise Exception(f"ImgBB API error: {response.status_code} - {response.text}")

    def _upload_to_catbox(self, image_data: str) -> str:
        """Upload to Catbox.moe file sharing service (200MB limit, no API key needed)"""
        import base64
        
        print(f"🔄 [Catbox] Uploading to Catbox.moe (200MB limit, no expiration)...")
        
        image_bytes = base64.b64decode(image_data)
        
        # Catbox.moe API endpoint
        response = requests.post(
            'https://catbox.moe/user/api.php',
            data={
                'reqtype': 'fileupload'
            },
            files={
                'fileToUpload': ('image.png', image_bytes, 'image/png')
            },
            timeout=60  # Longer timeout for large files
        )
        
        print(f"📡 [Catbox] Response status: {response.status_code}")
        
        if response.status_code == 200:
            url = response.text.strip()
            # Catbox returns just the URL on success
            if url.startswith('https://files.catbox.moe/'):
                print(f"✅ [Catbox] Upload successful: {url}")
                return url
            else:
                # If it's an error message, it won't start with https://
                print(f"❌ [Catbox] Upload failed: {url}")
                raise Exception(f"Catbox upload failed: {url}")
        else:
            print(f"❌ [Catbox] HTTP error: {response.status_code} - {response.text}")
            raise Exception(f"Catbox upload failed with status {response.status_code}: {response.text}")

    def _upload_to_free_service(self, image_data: str) -> str:
        """Try uploading to free services that don't require API keys"""
        
        # Try Catbox.moe first - has 200MB limit, no API key needed
        try:
            return self._upload_to_catbox(image_data)
        except Exception as e1:
            print(f"Catbox.moe upload failed: {e1}")
            
            # Try 0x0.st - a simple file sharing service
            try:
                return self._upload_to_0x0st(image_data)
            except Exception as e2:
                print(f"0x0.st upload failed: {e2}")
                
                # Try PostImages as last resort
                try:
                    return self._upload_to_postimages(image_data)
                except Exception as e3:
                    print(f"PostImages upload failed: {e3}")
                    
                    raise Exception(
                        "All free hosting services failed. Please use ImgBB with an API key for reliable hosting."
                    )

    def _upload_to_0x0st(self, image_data: str) -> str:
        """Upload to 0x0.st file sharing service"""
        import base64
        
        image_bytes = base64.b64decode(image_data)
        
        response = requests.post(
            'https://0x0.st',
            files={'file': ('image.png', image_bytes, 'image/png')},
            timeout=30
        )
        
        if response.status_code == 200:
            url = response.text.strip()
            if url.startswith('https://0x0.st/'):
                return url
            else:
                raise Exception(f"Unexpected response format: {url}")
        else:
            raise Exception(f"Upload failed with status {response.status_code}: {response.text}")

    def _upload_to_postimages(self, image_data: str) -> str:
        """Upload to PostImages service"""
        import base64
        
        image_bytes = base64.b64decode(image_data)
        
        response = requests.post(
            'https://postimages.org/json/rr',
            files={'upload': ('image.png', image_bytes, 'image/png')},
            data={
                'token': '',
                'upload_session': '',
                'numfiles': '1',
                'gallery': '',
                'ui': 'json'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'url' in result:
                return result['url']
            else:
                raise Exception(f"No URL in response: {result}")
        else:
            raise Exception(f"Upload failed with status {response.status_code}")

    def _submit_upscale_job(self, endpoint: str, payload: Dict[str, Any], api_key: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"📡 [Upsampler API] Sending request to: https://upsampler.com/api/v1/{endpoint}")
        print(f"🔑 [Upsampler API] Using API key: {'*' * 8}[REDACTED]")
        
        response = requests.post(
            f"https://upsampler.com/api/v1/{endpoint}",
            json=payload,
            headers=headers
        )
        
        print(f"📡 [Upsampler API] Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ [Upsampler API] Request failed: {response.status_code} - {response.text}")
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        print(f"📋 [Upsampler API] Response: {result}")
        
        if result.get("status") == "FAILED":
            error = result.get('error')
            print(f"❌ [Upsampler API] Job failed immediately: {error}")
            raise Exception(f"Job failed: {error}")
        
        job_id = result["id"]
        credit_cost = result.get("creditCost", "unknown")
        print(f"✅ [Upsampler API] Job queued successfully! ID: {job_id}, Cost: {credit_cost} credits")
        
        return job_id

    def _wait_for_completion(self, job_id: str, api_key: str) -> Image.Image:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        pbar = ProgressBar(100)
        progress = 0
        check_count = 0
        
        while True:
            check_count += 1
            print(f"🔄 [Status Check #{check_count}] Checking job status...")
            
            response = requests.get(
                f"https://upsampler.com/api/v1/status/{job_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"❌ [Status Check] Failed: {response.status_code}")
                raise Exception(f"Status check failed: {response.status_code}")
            
            result = response.json()
            status = result.get("status")
            print(f"📊 [Status Check] Current status: {status}")
            
            if status == "SUCCESS":
                # Download the result image
                image_url = result["imageUrl"]  # Use full quality PNG
                compressed_url = result.get("compressedImageUrl")
                credit_cost = result.get("creditCost", "unknown")
                
                print(f"🎉 [Success] Upscaling completed!")
                print(f"💰 [Success] Credits used: {credit_cost}")
                print(f"🔗 [Success] Full quality URL: {image_url}")
                if compressed_url:
                    print(f"🔗 [Success] Compressed URL: {compressed_url}")
                
                print(f"⬇️ [Download] Downloading result image...")
                image_response = requests.get(image_url)
                
                if image_response.status_code != 200:
                    print(f"❌ [Download] Failed to download: {image_response.status_code}")
                    raise Exception(f"Failed to download result image: {image_response.status_code}")
                
                print(f"✅ [Download] Downloaded {len(image_response.content)} bytes")
                pbar.update(100)
                
                # Load image from response content
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.write(image_response.content)
                temp_file.close()
                
                try:
                    # Open and load the image into memory
                    with Image.open(temp_file.name) as img:
                        result_image = img.copy()  # Create a copy to ensure it's loaded into memory
                    print(f"🖼️ [Result] Final image: {result_image.size} ({result_image.mode})")
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                        print(f"🧹 [Cleanup] Temporary file deleted successfully")
                    except PermissionError as e:
                        print(f"⚠️ [Cleanup] Could not delete temp file (will be cleaned up by OS): {e}")
                        # Don't raise - the image is loaded, temp file will be cleaned up by OS
                        
                except Exception as e:
                    # If anything goes wrong, try to clean up
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                    raise e
                
                return result_image
                
            elif status == "FAILED":
                error = result.get('error')
                print(f"❌ [Failed] Upscaling failed: {error}")
                raise Exception(f"Upscaling failed: {error}")
            
            elif status in ["IN_PROGRESS", "IN_QUEUE"]:
                # Update progress bar
                progress = min(progress + 5, 95)
                pbar.update(progress)
                print(f"⏳ [Waiting] Status: {status}, waiting 60 seconds...")
                time.sleep(60)  # Wait 60 seconds before next check
            
            else:
                print(f"❓ [Unknown] Unexpected status: {status}")
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
                "imgbb_api_key": ("STRING", {"multiline": False, "default": ""}),
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
                detail: int, imgbb_api_key: str = "", description: str = "", should_enhance_faces: bool = False, 
                should_preserve_hands: bool = False, should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        # Reuse the same methods from SmartUpscale
        upscaler = UpsamplerSmartUpscale()
        pil_image = upscaler._tensor_to_pil(image)
        
        # Get ImgBB API key from parameter or environment variable
        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        image_url = upscaler._upload_image_temp(pil_image, hosting_api_key)
        
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
                "imgbb_api_key": ("STRING", {"multiline": False, "default": ""}),
                "should_enhance_faces": ("BOOLEAN", {"default": False}),
                "should_preserve_blur": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"

    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, imgbb_api_key: str = "", should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False) -> Tuple[torch.Tensor]:
        
        # Reuse the same methods from SmartUpscale
        upscaler = UpsamplerSmartUpscale()
        pil_image = upscaler._tensor_to_pil(image)
        
        # Get ImgBB API key from parameter or environment variable
        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        image_url = upscaler._upload_image_temp(pil_image, hosting_api_key)
        
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

print("✅ ImgBB-only Upsampler nodes loaded successfully")