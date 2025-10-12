import os
import requests
import time
import tempfile
from typing import Tuple, Optional, Dict, Any, Callable, List
from urllib.parse import urlparse
from PIL import Image
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import folder_paths
from comfy.utils import ProgressBar




class UpsamplerJobError(Exception):
    """Custom exception used to mark whether an Upsampler job can be retried."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class UpsamplerSmartUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or tile tensor to upscale (values in [0,1])."}),
                "api_key": ("STRING", {"multiline": False, "tooltip": "Upsampler API key used to authenticate requests."}),
                "input_image_type": (["realism", "anime", "universal"], {"tooltip": "Choose the preset that best matches the source image style."}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Overall scale factor applied by the Upsampler service."}),
                "global_creativity": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Controls how strongly the service reimagines the image."}),
                "detail": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Detail enhancement intensity for the upscaled result."}),
            },
            "optional": {
                "imgbb_api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional ImgBB API key for authenticated temporary hosting."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional text prompt sent with the upscale request."}),
                "should_enhance_faces": ("BOOLEAN", {"default": False, "tooltip": "Enable Upsampler face enhancement for portraits."}),
                "should_preserve_blur": ("BOOLEAN", {"default": False, "tooltip": "Maintain natural blur instead of sharpening every region."}),
                "max_parallel_jobs": ("INT", {"default": 1, "min": 1, "max": 15, "tooltip": "Maximum parallel Upsampler jobs per node (1-15). Respects UPSAMPLER_MAX_CONCURRENCY and API rate limits."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"


    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, global_creativity: int, detail: int,
                imgbb_api_key: str = "", description: str = "", should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False, max_parallel_jobs: int = 1) -> Tuple[torch.Tensor]:
        print(f"?? [Upsampler] Starting Smart Upscale process...")
        print(f"?? [Upsampler] Input image shape: {image.shape}")
        print(f"??? [Upsampler] Parameters - Type: {input_image_type}, Factor: {upscale_factor}, Creativity: {global_creativity}, Detail: {detail}")

        image_batch = self._prepare_image_batch(image)
        total_tiles = len(image_batch)
        if total_tiles > 1:
            print(f"?? [Upsampler] Detected batched input: {total_tiles} tiles will be processed.")

        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        print(f"?? [Upsampler] Image hosting: {'ImgBB (with API key)' if hosting_api_key else 'Free services (no API key)'}")

        payload_builder = lambda image_url: {
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




        concurrency = self._resolve_concurrency(max_parallel_jobs, total_tiles)
        if total_tiles > 1:
            print(f"?? [Upsampler] Processing {total_tiles} tiles with up to {concurrency} concurrent job(s).")

        outputs = self._process_tiles(
            image_batch,
            api_key=api_key,
            hosting_api_key=hosting_api_key,
            payload_builder=payload_builder,
            job_type="smart-upscale",
            max_concurrent_jobs=concurrency,
        )

        result_tensor = torch.cat(outputs, dim=0)
        if total_tiles > 1:
            print(f"?? [Upsampler] Batched upscaling complete. Combined tensor shape: {result_tensor.shape}")

        print(f"? [Upsampler] Smart Upscale process completed successfully!")
        return (result_tensor,)


    def _prepare_image_batch(self, image: torch.Tensor) -> List[torch.Tensor]:
        if image.ndim == 3:
            return [image.unsqueeze(0)]
        if image.ndim == 4:
            return [image[i:i + 1] for i in range(image.shape[0])]
        raise ValueError(f"Unsupported image tensor shape: {image.shape}")


    def _upscale_single_image(
        self,
        tile: torch.Tensor,
        *,
        api_key: str,
        hosting_api_key: str,
        payload_builder: Callable[[str], Dict[str, Any]],
        job_type: str,
        tile_index: int,
        total_tiles: int,
    ) -> torch.Tensor:
        tile_label = f"{tile_index}/{total_tiles}" if total_tiles > 1 else "1/1"

        pil_image = self._tensor_to_pil(tile)
        if total_tiles > 1:
            print(f"??? [Upsampler] Tile {tile_label}: Converted to PIL image: {pil_image.size} ({pil_image.mode})")
        else:
            print(f"??? [Upsampler] Converted to PIL image: {pil_image.size} ({pil_image.mode})")

        max_retries_env = os.getenv("UPSAMPLER_MAX_RETRIES")
        try:
            max_retries = int(max_retries_env) if max_retries_env is not None else 2
        except ValueError:
            max_retries = 2
        max_retries = max(0, max_retries)

        retry_delay_env = os.getenv("UPSAMPLER_RETRY_DELAY")
        try:
            retry_delay = float(retry_delay_env) if retry_delay_env is not None else 10.0
        except ValueError:
            retry_delay = 10.0
        retry_delay = max(0.0, retry_delay)

        max_attempts = max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            attempt_label = f"{attempt}/{max_attempts}"
            if attempt > 1:
                last_error_message = str(last_error) if last_error else "unknown error"
                print(
                    f"?? [Upsampler] Tile {tile_label}: Retrying after failure ({attempt_label}). Last error: {last_error_message}"
                )

            upload_action = (
                "Uploading image to hosting service"
                if attempt == 1
                else f"Re-uploading image for retry attempt {attempt_label}"
            )
            print(f"?? [Upsampler] Tile {tile_label}: {upload_action}...")
            image_url = self._upload_image_temp(pil_image, hosting_api_key)
            print(f"? [Upsampler] Tile {tile_label}: Image uploaded successfully: {image_url}")

            payload = payload_builder(image_url)
            self._ensure_within_api_limits(tile, tile_label, payload)

            print(f"?? [Upsampler] Tile {tile_label}: Submitting job to Upsampler API (attempt {attempt_label})...")
            try:
                job_id = self._submit_upscale_job(job_type, payload, api_key)
                print(f"? [Upsampler] Tile {tile_label}: Job submitted successfully! Job ID: {job_id}")
                print(f"? [Upsampler] Tile {tile_label}: Waiting for upscaling to complete...")
                result_image = self._wait_for_completion(
                    job_id,
                    api_key,
                    tile_label=tile_label,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            except UpsamplerJobError as exc:
                last_error = exc
                retryable = exc.retryable and attempt < max_attempts
                print(f"? [Upsampler] Tile {tile_label}: Attempt {attempt_label} failed: {exc}")
                if retryable:
                    if retry_delay:
                        print(
                            f"?? [Upsampler] Tile {tile_label}: Waiting {retry_delay:.0f} seconds before retrying..."
                        )
                        time.sleep(retry_delay)
                    continue
                raise

            result_tensor = self._pil_to_tensor(result_image)
            print(f"?? [Upsampler] Tile {tile_label}: Final tensor shape: {result_tensor.shape}")

            self._validate_output(result_tensor, result_image)

            return result_tensor

        raise UpsamplerJobError(f"Tile {tile_label} failed after {max_attempts} attempts", retryable=False)



    def _resolve_concurrency(self, requested: Optional[int], total_tiles: int) -> int:
        env_value = os.getenv("UPSAMPLER_MAX_CONCURRENCY")
        concurrency = requested if requested is not None else 1
        if env_value is not None:
            try:
                concurrency = int(env_value)
                print(f"?? [Upsampler] UPSAMPLER_MAX_CONCURRENCY override detected: {env_value}")
            except ValueError:
                print(f"? [Upsampler] Invalid UPSAMPLER_MAX_CONCURRENCY value '{env_value}', ignoring override.")
        concurrency = max(1, min(concurrency, 15))
        if total_tiles > 0:
            concurrency = min(concurrency, total_tiles)
        return concurrency

    def _process_tiles(
        self,
        image_batch: List[torch.Tensor],
        *,
        api_key: str,
        hosting_api_key: str,
        payload_builder: Callable[[str], Dict[str, Any]],
        job_type: str,
        max_concurrent_jobs: int,
    ) -> List[torch.Tensor]:
        total_tiles = len(image_batch)
        if total_tiles == 0:
            return []

        if max_concurrent_jobs <= 1 or total_tiles == 1:
            outputs: List[torch.Tensor] = []
            for idx, tile in enumerate(image_batch, start=1):
                outputs.append(
                    self._upscale_single_image(
                        tile,
                        api_key=api_key,
                        hosting_api_key=hosting_api_key,
                        payload_builder=payload_builder,
                        job_type=job_type,
                        tile_index=idx,
                        total_tiles=total_tiles,
                    )
                )
            return outputs

        results: List[Optional[torch.Tensor]] = [None] * total_tiles

        def run_tile(idx: int, tile_tensor: torch.Tensor):
            tensor = self._upscale_single_image(
                tile_tensor,
                api_key=api_key,
                hosting_api_key=hosting_api_key,
                payload_builder=payload_builder,
                job_type=job_type,
                tile_index=idx,
                total_tiles=total_tiles,
            )
            return idx, tensor

        with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
            futures = [
                executor.submit(run_tile, idx, tile)
                for idx, tile in enumerate(image_batch, start=1)
            ]
            try:
                for future in as_completed(futures):
                    idx, tensor = future.result()
                    results[idx - 1] = tensor
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        return [tensor for tensor in results if tensor is not None]

    def _ensure_within_api_limits(self, tile: torch.Tensor, tile_label: str, payload: Dict[str, Any]):
        upscale_factor = float(payload.get("input", {}).get("upscaleFactor", 1.0))
        tile_height = tile.shape[1]
        tile_width = tile.shape[2]
        projected_pixels = tile_height * tile_width * (upscale_factor ** 2)
        max_pixels = 50_000_000

        if projected_pixels > max_pixels:
            limit_factor = (max_pixels / (tile_height * tile_width)) ** 0.5
            message = (
                f"Tile {tile_label} would produce {projected_pixels:.0f} pixels, exceeding the 50M limit. "
                f"Reduce the upscale factor to <= {limit_factor:.2f} or use smaller tiles."
            )
            raise Exception(message)


    def _is_retryable_failure(self, error_message: Optional[str]) -> bool:
        if not error_message:
            return True
        lowered = error_message.lower()
        non_retryable_markers = [
            "invalid api key",
            "insufficient credit",
            "insufficient credits",
            "unauthorized",
            "unsupported",
            "not allowed",
            "exceed",
            "too large",
        ]
        return not any(marker in lowered for marker in non_retryable_markers)

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
            try:
                return self._upload_to_imgbb(image_data, imgbb_api_key)
            except Exception as exc:
                print(f"?? [ImgBB] Upload failed ({exc}). Falling back to free services...")
        
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
        
        try:
            response = requests.post(
                "https://api.imgbb.com/1/upload",
                data={
                    "key": api_key,
                    "image": image_data,
                    "expiration": 3600  # 1 hour expiration
                },
                timeout=30
            )
        except requests.exceptions.RequestException as exc:
            print(f"? [ImgBB] Request error: {exc}")
            raise Exception(f"ImgBB request failed: {exc}")
        
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

        print(f"?? [Upsampler API] Sending request to: https://upsampler.com/api/v1/{endpoint}")
        print(f"?? [Upsampler API] Using API key: {'*' * 8}[REDACTED]")

        try:
            response = requests.post(
                f"https://upsampler.com/api/v1/{endpoint}",
                json=payload,
                headers=headers
            )
        except requests.exceptions.RequestException as exc:
            print(f"? [Upsampler API] Request error: {exc}")
            raise UpsamplerJobError(f"API request failed: {exc}", retryable=True) from exc

        print(f"?? [Upsampler API] Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"? [Upsampler API] Request failed: {response.status_code} - {response.text}")
            retryable = response.status_code >= 500
            raise UpsamplerJobError(
                f"API request failed: {response.status_code} - {response.text}",
                retryable=retryable,
            )

        result = response.json()
        print(f"?? [Upsampler API] Response: {result}")

        if result.get("status") == "FAILED":
            error = result.get('error')
            print(f"? [Upsampler API] Job failed immediately: {error}")
            retryable = self._is_retryable_failure(error)
            raise UpsamplerJobError(f"Job failed: {error}", retryable=retryable)

        job_id = result["id"]
        credit_cost = result.get("creditCost", "unknown")
        print(f"? [Upsampler API] Job queued successfully! ID: {job_id}, Cost: {credit_cost} credits")

        return job_id


    def _wait_for_completion(
        self,
        job_id: str,
        api_key: str,
        *,
        tile_label: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> Image.Image:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        pbar = ProgressBar(100)
        progress = 0
        check_count = 0
        tile_prefix = f"Tile {tile_label}: " if tile_label else ""
        attempt_suffix = f" (attempt {attempt}/{max_attempts})" if max_attempts > 1 else ""

        while True:
            check_count += 1
            print(f"?? [Status Check #{check_count}] {tile_prefix}Checking job status{attempt_suffix}...")

            try:
                response = requests.get(
                    f"https://upsampler.com/api/v1/status/{job_id}",
                    headers=headers
                )
            except requests.exceptions.RequestException as exc:
                print(f"? [Status Check] Request error: {exc}")
                raise UpsamplerJobError(f"Status check request failed: {exc}", retryable=True) from exc

            if response.status_code != 200:
                print(f"? [Status Check] Failed: {response.status_code}")
                retryable = response.status_code >= 500
                raise UpsamplerJobError(
                    f"Status check failed: {response.status_code}",
                    retryable=retryable,
                )

            result = response.json()
            status = result.get("status")
            print(f"?? [Status Check] {tile_prefix}Current status: {status}")

            if status == "SUCCESS":
                image_url = result["imageUrl"]
                compressed_url = result.get("compressedImageUrl")
                credit_cost = result.get("creditCost", "unknown")

                print(f"?? [Success] {tile_prefix}Upscaling completed!")
                print(f"?? [Success] {tile_prefix}Credits used: {credit_cost}")
                print(f"?? [Success] {tile_prefix}Full quality URL: {image_url}")
                if compressed_url:
                    print(f"?? [Success] {tile_prefix}Compressed URL: {compressed_url}")

                print(f"?? [Download] {tile_prefix}Downloading result image...")
                try:
                    image_response = requests.get(image_url)
                except requests.exceptions.RequestException as exc:
                    print(f"? [Download] Request error: {exc}")
                    raise UpsamplerJobError(f"Failed to download result image: {exc}", retryable=True) from exc

                if image_response.status_code != 200:
                    print(f"? [Download] Failed to download: {image_response.status_code}")
                    retryable = image_response.status_code >= 500
                    raise UpsamplerJobError(
                        f"Failed to download result image: {image_response.status_code}",
                        retryable=retryable,
                    )

                print(f"? [Download] Downloaded {len(image_response.content)} bytes")
                pbar.update(100)

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.write(image_response.content)
                temp_file.close()

                try:
                    with Image.open(temp_file.name) as img:
                        result_image = img.copy()
                    print(f"??? [Result] {tile_prefix}Final image: {result_image.size} ({result_image.mode})")

                    try:
                        os.unlink(temp_file.name)
                        print(f"?? [Cleanup] Temporary file deleted successfully")
                    except PermissionError as e:
                        print(f"?? [Cleanup] Could not delete temp file (will be cleaned up by OS): {e}")
                except Exception as e:
                    try:
                        os.unlink(temp_file.name)
                    except Exception:
                        pass
                    raise UpsamplerJobError(f"Failed to load result image: {e}", retryable=True) from e

                return result_image

            elif status == "FAILED":
                error = result.get('error')
                print(f"? [Failed] {tile_prefix}Upscaling failed: {error}")
                retryable = self._is_retryable_failure(error)
                raise UpsamplerJobError(f"Upscaling failed: {error}", retryable=retryable)

            elif status in ["IN_PROGRESS", "IN_QUEUE"]:
                progress = min(progress + 5, 95)
                pbar.update(progress)
                print(f"? [Waiting] {tile_prefix}Status: {status}, waiting 60 seconds...")
                time.sleep(60)

            else:
                print(f"? [Unknown] {tile_prefix}Unexpected status: {status}")
                raise UpsamplerJobError(f"Unknown status: {status}", retryable=True)



class UpsamplerDynamicUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or tile tensor to upscale (values in [0,1])."}),
                "api_key": ("STRING", {"multiline": False, "tooltip": "Upsampler API key used to authenticate requests."}),
                "input_image_type": (["realism", "anime", "universal"], {"tooltip": "Choose the preset that best matches the source image style."}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Overall scale factor applied by the Upsampler service."}),
                "global_creativity": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Controls how strongly the service reimagines the image."}),
                "resemblance": ("INT", {"default": 8, "min": 0, "max": 10, "tooltip": "Adjust how closely the output should match the input."}),
                "detail": ("INT", {"default": 5, "min": 0, "max": 10, "tooltip": "Detail enhancement intensity for the upscaled result."}),
            },
            "optional": {
                "imgbb_api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional ImgBB API key for authenticated temporary hosting."}),
                "description": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional text prompt sent with the upscale request."}),
                "should_enhance_faces": ("BOOLEAN", {"default": False, "tooltip": "Enable Upsampler face enhancement for portraits."}),
                "should_preserve_hands": ("BOOLEAN", {"default": False, "tooltip": "Try to maintain accurate hand details when present."}),
                "should_preserve_blur": ("BOOLEAN", {"default": False, "tooltip": "Maintain natural blur instead of sharpening every region."}),
                "max_parallel_jobs": ("INT", {"default": 1, "min": 1, "max": 15, "tooltip": "Maximum parallel Upsampler jobs per node (1-15). Respects UPSAMPLER_MAX_CONCURRENCY and API rate limits."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"


    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, global_creativity: int, resemblance: int, 
                detail: int, imgbb_api_key: str = "", description: str = "", should_enhance_faces: bool = False, 
                should_preserve_hands: bool = False, should_preserve_blur: bool = False, max_parallel_jobs: int = 1) -> Tuple[torch.Tensor]:
        upscaler = UpsamplerSmartUpscale()
        image_batch = upscaler._prepare_image_batch(image)
        total_tiles = len(image_batch)
        if total_tiles > 1:
            print(f"?? [Upsampler] Detected batched input: {total_tiles} tiles will be processed.")

        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        print(f"?? [Upsampler] Image hosting: {'ImgBB (with API key)' if hosting_api_key else 'Free services (no API key)'}")

        payload_builder = lambda image_url: {
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


        concurrency = upscaler._resolve_concurrency(max_parallel_jobs, total_tiles)
        if total_tiles > 1:
            print(f"?? [Upsampler] Processing {total_tiles} tiles with up to {concurrency} concurrent job(s).")


        outputs = upscaler._process_tiles(
            image_batch,
            api_key=api_key,
            hosting_api_key=hosting_api_key,
            payload_builder=payload_builder,
            job_type="dynamic-upscale",
            max_concurrent_jobs=concurrency,
        )

        result_tensor = torch.cat(outputs, dim=0)
        if total_tiles > 1:
            print(f"?? [Upsampler] Batched upscaling complete. Combined tensor shape: {result_tensor.shape}")

        print(f"? [Upsampler] Dynamic Upscale process completed successfully!")
        return (result_tensor,)


class UpsamplerPreciseUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or tile tensor to upscale (values in [0,1])."}),
                "api_key": ("STRING", {"multiline": False, "tooltip": "Upsampler API key used to authenticate requests."}),
                "input_image_type": (["realism", "anime", "universal"], {"tooltip": "Choose the preset that best matches the source image style."}),
                "upscale_factor": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.1, "tooltip": "Overall scale factor applied by the Upsampler service."}),
            },
            "optional": {
                "imgbb_api_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Optional ImgBB API key for authenticated temporary hosting."}),
                "should_enhance_faces": ("BOOLEAN", {"default": False, "tooltip": "Enable Upsampler face enhancement for portraits."}),
                "should_preserve_blur": ("BOOLEAN", {"default": False, "tooltip": "Maintain natural blur instead of sharpening every region."}),
                "max_parallel_jobs": ("INT", {"default": 1, "min": 1, "max": 15, "tooltip": "Maximum parallel Upsampler jobs per node (1-15). Respects UPSAMPLER_MAX_CONCURRENCY and API rate limits."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "Upsampler"


    def upscale(self, image: torch.Tensor, api_key: str, input_image_type: str, 
                upscale_factor: float, imgbb_api_key: str = "", should_enhance_faces: bool = False, 
                should_preserve_blur: bool = False, max_parallel_jobs: int = 1) -> Tuple[torch.Tensor]:
        upscaler = UpsamplerSmartUpscale()
        image_batch = upscaler._prepare_image_batch(image)
        total_tiles = len(image_batch)
        if total_tiles > 1:
            print(f"?? [Upsampler] Detected batched input: {total_tiles} tiles will be processed.")

        hosting_api_key = imgbb_api_key or os.getenv('IMGBB_API_KEY', '')
        print(f"?? [Upsampler] Image hosting: {'ImgBB (with API key)' if hosting_api_key else 'Free services (no API key)'}")

        payload_builder = lambda image_url: {
            "input": {
                "imageUrl": image_url,
                "inputImageType": input_image_type,
                "upscaleFactor": upscale_factor,
                "shouldEnhanceFaces": should_enhance_faces,
                "shouldPreserveBlur": should_preserve_blur
            }
        }


        concurrency = upscaler._resolve_concurrency(max_parallel_jobs, total_tiles)
        if total_tiles > 1:
            print(f"?? [Upsampler] Processing {total_tiles} tiles with up to {concurrency} concurrent job(s).")


        outputs = upscaler._process_tiles(
            image_batch,
            api_key=api_key,
            hosting_api_key=hosting_api_key,
            payload_builder=payload_builder,
            job_type="precise-upscale",
            max_concurrent_jobs=concurrency,
        )

        result_tensor = torch.cat(outputs, dim=0)
        if total_tiles > 1:
            print(f"?? [Upsampler] Batched upscaling complete. Combined tensor shape: {result_tensor.shape}")

        print(f"? [Upsampler] Precise Upscale process completed successfully!")
        return (result_tensor,)


print("✅ ImgBB-only Upsampler nodes loaded successfully")



