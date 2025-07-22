from .nodes import UpsamplerSmartUpscale, UpsamplerDynamicUpscale, UpsamplerPreciseUpscale

NODE_CLASS_MAPPINGS = {
    "Upsampler Smart Upscale": UpsamplerSmartUpscale,
    "Upsampler Dynamic Upscale": UpsamplerDynamicUpscale,
    "Upsampler Precise Upscale": UpsamplerPreciseUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Upsampler Smart Upscale": "🔍 Upsampler Smart Upscale",
    "Upsampler Dynamic Upscale": "⚡ Upsampler Dynamic Upscale", 
    "Upsampler Precise Upscale": "🎯 Upsampler Precise Upscale",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']