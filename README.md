# ComfyUI Upsampler Nodes

ComfyUI custom nodes for integrating with the Upsampler API to enhance and upscale images using AI.

## Features

Three different upscaling methods:

1. **Smart Upscale** - Enhances images by intelligently regenerating at a higher resolution
2. **Dynamic Upscale** - Preserves the overall composition while forcing new details  
3. **Precise Upscale** - Upscales and sharpens while completely preserving the original structure

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:
   ```
   cd ComfyUI/custom_nodes
   git clone <this-repo-url> comfyui-upsampler
   ```

2. Install dependencies:
   ```
   pip install -r comfyui-upsampler/requirements.txt
   ```

3. Restart ComfyUI

## Setup Required

### 1. Get Upsampler API Key

1. Visit https://upsampler.com/
2. Create an account and generate an API key
3. Add credits to your account for processing

### 2. Implement Image Hosting

**IMPORTANT**: Before using these nodes, you must implement the `_upload_image_temp()` method in `nodes.py` to upload images to a publicly accessible URL.

The Upsampler API requires images to be accessible via public URLs. You have several options:

#### Option A: Use ImgBB (Recommended)
1. Get a free API key from https://api.imgbb.com/
2. Uncomment and modify the ImgBB implementation in `_upload_image_temp()`

#### Option B: Use Your Own Server
Upload images to your own web server or CDN that provides public URLs.

#### Option C: Use Other Services
Implement integration with services like Imgur, Cloudinary, or AWS S3.

## Usage

1. Add one of the Upsampler nodes to your ComfyUI workflow:
   - 🔍 Upsampler Smart Upscale
   - ⚡ Upsampler Dynamic Upscale  
   - 🎯 Upsampler Precise Upscale

2. Connect an IMAGE input to the node

3. Configure the parameters:
   - **API Key**: Your Upsampler API key
   - **Input Image Type**: Choose "realism", "anime", or "universal"
   - **Upscale Factor**: How much to upscale (1.0-4.0 for Smart/Dynamic, 1.0-16.0 for Precise)
   - Additional parameters specific to each method

4. The node will:
   - Upload your image to a public URL
   - Submit the upscaling job to Upsampler API
   - Poll for completion with progress updates
   - Download and return the upscaled image

## Node Parameters

### Smart Upscale
- `global_creativity` (0-10): How much the AI can change the original image
- `detail` (0-10): Level of detail enhancement
- `description`: Text prompt to guide the upscaling process
- `should_enhance_faces`: Better preserve and enhance facial features
- `should_preserve_blur`: Preserve existing blur in the image

### Dynamic Upscale  
- `global_creativity` (0-10): How much the AI can change the original image
- `resemblance` (0-10): How closely to adhere to original image structure
- `detail` (0-10): Level of detail enhancement
- `description`: Text prompt to guide the upscaling process
- `should_enhance_faces`: Better preserve and enhance facial features
- `should_preserve_hands`: Better preserve hand structures
- `should_preserve_blur`: Preserve existing blur in the image

### Precise Upscale
- `should_enhance_faces`: Apply face restoration techniques
- `should_preserve_blur`: Preserve existing blur in the image

## Pricing

Credits are consumed based on output image size:
- **Smart/Dynamic Upscale**: 1 credit per megapixel (minimum 1 credit)
- **Precise Upscale**: 1 credit per 4 megapixels (minimum 1 credit)

## Troubleshooting

1. **"Image hosting not implemented" error**: You need to implement the `_upload_image_temp()` method
2. **API authentication errors**: Check your API key is correct and has credits
3. **Rate limit errors**: The API has limits of 60 requests/minute and 1800 requests/hour
4. **Job failures**: Check the error message - common issues include invalid parameters or insufficient credits

## API Documentation

For full API documentation, visit: https://upsampler.com/api

## License

This custom node implementation is provided as-is. Please refer to Upsampler's terms of service for API usage.