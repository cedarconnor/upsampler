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

### 2. Image Hosting Setup

The Upsampler API requires images to be accessible via public URLs. The nodes now include **automatic image hosting** with multiple options:

#### Option A: ImgBB (Recommended - Free & Reliable)

1. **Get a free API key**:
   - Visit https://api.imgbb.com/
   - Sign up and generate an API key
   - Free tier: Up to 100 images/day

2. **Configure the API key** (choose one method):
   
   **Method 1: Node Parameter**
   - Add your ImgBB API key to the "imgbb_api_key" field in each node
   
   **Method 2: Environment Variable**
   - Set environment variable: `IMGBB_API_KEY=your_api_key_here`
   - Windows: `set IMGBB_API_KEY=your_api_key_here`
   - Linux/Mac: `export IMGBB_API_KEY=your_api_key_here`

#### Option B: Free Services (Automatic Fallback)

If no ImgBB API key is provided, the nodes will automatically try:
- **0x0.st**: Simple file sharing service
- **PostImages**: Image hosting service

**Note**: Free services may be less reliable and have usage limits.

#### Option C: Your Own Server
For production use, implement your own hosting solution by modifying the `_upload_image_temp()` method in `nodes.py`.

## Usage

1. Add one of the Upsampler nodes to your ComfyUI workflow:
   - 🔍 Upsampler Smart Upscale
   - ⚡ Upsampler Dynamic Upscale  
   - 🎯 Upsampler Precise Upscale

2. Connect an IMAGE input to the node

3. Configure the parameters:
   - **API Key**: Your Upsampler API key
   - **ImgBB API Key** (optional): Your ImgBB API key for reliable image hosting
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
- `imgbb_api_key`: ImgBB API key for reliable image hosting (optional)
- `global_creativity` (0-10): How much the AI can change the original image
- `detail` (0-10): Level of detail enhancement
- `description`: Text prompt to guide the upscaling process
- `should_enhance_faces`: Better preserve and enhance facial features
- `should_preserve_blur`: Preserve existing blur in the image

### Dynamic Upscale  
- `imgbb_api_key`: ImgBB API key for reliable image hosting (optional)
- `global_creativity` (0-10): How much the AI can change the original image
- `resemblance` (0-10): How closely to adhere to original image structure
- `detail` (0-10): Level of detail enhancement
- `description`: Text prompt to guide the upscaling process
- `should_enhance_faces`: Better preserve and enhance facial features
- `should_preserve_hands`: Better preserve hand structures
- `should_preserve_blur`: Preserve existing blur in the image

### Precise Upscale
- `imgbb_api_key`: ImgBB API key for reliable image hosting (optional)
- `should_enhance_faces`: Apply face restoration techniques
- `should_preserve_blur`: Preserve existing blur in the image

## Pricing

Credits are consumed based on output image size:
- **Smart/Dynamic Upscale**: 1 credit per megapixel (minimum 1 credit)
- **Precise Upscale**: 1 credit per 4 megapixels (minimum 1 credit)

## Troubleshooting

1. **Image upload errors**: 
   - Get an ImgBB API key for reliable hosting
   - Check your internet connection
   - Try again if free services are temporarily unavailable

2. **"All free hosting services failed"**: 
   - Use ImgBB with an API key (recommended)
   - Set the `IMGBB_API_KEY` environment variable
   - Or add the key to the node parameter

3. **API authentication errors**: 
   - Check your Upsampler API key is correct and has credits
   - Verify account status at https://upsampler.com/

4. **Rate limit errors**: 
   - The API has limits of 60 requests/minute and 1800 requests/hour
   - Wait before retrying

5. **Job failures**: 
   - Check the error message for specific issues
   - Common issues: invalid parameters or insufficient credits
   - Verify image format and size constraints

## API Documentation

For full API documentation, visit: https://upsampler.com/api

## License

This custom node implementation is provided as-is. Please refer to Upsampler's terms of service for API usage.