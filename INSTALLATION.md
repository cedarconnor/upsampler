# Installation Guide

## Quick Start (ImgBB Only)

If you only want to use ImgBB (supports images up to 32MB):

1. **No additional installation needed** - the node will work out of the box
2. Get an ImgBB API key from https://api.imgbb.com/
3. Use the `imgbb_api_key` parameter in the node

## Full Installation (ImgBB + Google Drive)

For large image support (>32MB) via Google Drive:

### Install Dependencies

```bash
pip install google-api-python-client google-auth
```

**Or if you're using ComfyUI's Python environment:**

```bash
# Windows
path\to\comfyui\python_embeded\python.exe -m pip install google-api-python-client google-auth

# Linux/Mac
/path/to/comfyui/venv/bin/pip install google-api-python-client google-auth
```

### Setup Google Drive

Follow the detailed setup guide in [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md)

## Troubleshooting

### "Missing Node Types" Error

This usually means the Google API dependencies aren't installed:

1. **Check ComfyUI console** for error messages
2. **Install dependencies** using the commands above
3. **Restart ComfyUI** completely
4. **Look for these messages** in console:
   - `✅ [Upsampler] Google Drive integration available` (good)
   - `⚠️ [Upsampler] Google Drive integration disabled` (install dependencies)

### Still Having Issues?

1. Check that you're installing in the correct Python environment
2. Make sure ComfyUI can find the installed packages
3. Try restarting ComfyUI after installation
4. Check the console for detailed error messages