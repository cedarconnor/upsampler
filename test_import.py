#!/usr/bin/env python3

print("Testing ComfyUI Upsampler node imports...")

try:
    print("1. Testing basic imports...")
    import os
    import requests
    import time
    import tempfile
    import json
    from typing import Tuple, Optional, Dict, Any
    from urllib.parse import urlparse
    from PIL import Image
    import torch
    import numpy as np
    print("   ✅ Basic imports successful")
except Exception as e:
    print(f"   ❌ Basic imports failed: {e}")
    exit(1)

try:
    print("2. Testing ComfyUI imports...")
    import folder_paths
    from comfy.utils import ProgressBar
    print("   ✅ ComfyUI imports successful")
except Exception as e:
    print(f"   ❌ ComfyUI imports failed: {e}")
    print("   ℹ️  This is normal if not running from ComfyUI environment")

try:
    print("3. Testing Google Drive imports...")
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.service_account import Credentials
    print("   ✅ Google Drive imports successful")
except ImportError as e:
    print(f"   ⚠️  Google Drive imports failed: {e}")
    print("   ℹ️  This is expected if Google API libraries aren't installed")

try:
    print("4. Testing node class imports...")
    from nodes import UpsamplerSmartUpscale, UpsamplerDynamicUpscale, UpsamplerPreciseUpscale
    print("   ✅ Node class imports successful")
    
    print("5. Testing node class instantiation...")
    smart = UpsamplerSmartUpscale()
    dynamic = UpsamplerDynamicUpscale()
    precise = UpsamplerPreciseUpscale()
    print("   ✅ Node class instantiation successful")
    
    print("6. Testing INPUT_TYPES methods...")
    smart_types = smart.INPUT_TYPES()
    dynamic_types = dynamic.INPUT_TYPES()
    precise_types = precise.INPUT_TYPES()
    print("   ✅ INPUT_TYPES methods successful")
    
    print("\n🎉 All tests passed! Node should load properly in ComfyUI.")
    
except Exception as e:
    print(f"   ❌ Node import/instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)