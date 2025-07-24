# Google Drive Setup for ComfyUI Upsampler

This guide explains how to set up Google Drive as an upload method for large images (>30MB) that exceed ImgBB's free limit.

## Prerequisites

Install the required Google API libraries:
```bash
pip install google-api-python-client google-auth
```

## Setup Steps

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API for your project

### 2. Create Service Account
1. Go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a name (e.g., "upsampler-drive")
4. Click "Create and Continue"
5. Skip role assignment (click "Continue")
6. Click "Done"

### 3. Generate Credentials
1. Click on your created service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select "JSON" format
5. Download the JSON file
6. Save it securely (e.g., `~/google-drive-creds.json`)

### 4. Configure the Node

#### Method 1: Use node parameters (Recommended)
- Set `google_drive_creds_path` to your JSON file path
- Set `google_drive_folder_id` to your folder ID (optional)
- Set `upload_method` to "google_drive" or "auto"

#### Method 2: Use environment variables
Set these environment variables:
```bash
export GOOGLE_DRIVE_CREDS_PATH="/path/to/your/credentials.json"
export GOOGLE_DRIVE_FOLDER_ID="your_folder_id"  # Optional - can also be set in node
```

**Note:** Node parameters take priority over environment variables.

## Usage

### Upload Method Options:
- **"auto"** (default): Automatically chooses based on image size and available credentials
- **"google_drive"**: Forces Google Drive upload
- **"imgbb"**: Forces ImgBB upload (requires API key)

### Auto Selection Logic:
- Images >30MB: Prefers Google Drive if credentials available
- Images ≤30MB: Prefers ImgBB if API key available
- Falls back to free services if no credentials

## Optional: Specify Upload Folder

To upload to a specific Google Drive folder:

### Method 1: Using Node Parameter (Easier)
1. Create or navigate to the desired folder in Google Drive
2. Copy the folder ID from the URL (the long string after `/folders/`)
   - Example URL: `https://drive.google.com/drive/folders/1ABC123DEF456GHI789JKL`
   - Folder ID: `1ABC123DEF456GHI789JKL`
3. Paste this ID into the `google_drive_folder_id` field in the node

### Method 2: Using Environment Variable
1. Follow steps 1-2 above to get the folder ID
2. Set the `GOOGLE_DRIVE_FOLDER_ID` environment variable to this ID

**Note:** The node parameter takes priority over the environment variable.

## Security Notes

- Keep your service account JSON file secure
- Don't commit credentials to version control
- The service account only has access to files it creates
- Uploaded files are made publicly readable (required for Upsampler API)

## Troubleshooting

### "Credentials file not found"
- Check the file path is correct
- Ensure the file exists and is readable
- Use absolute paths

### "Permission denied" 
- Verify the service account has Drive API access
- Check the JSON credentials file is valid

### "Module not found"
- Install required packages: `pip install google-api-python-client google-auth`

## Quick Setup Summary

Once you've completed the setup, your node will have these new fields:

1. **`google_drive_creds_path`**: Path to your downloaded JSON credentials file
2. **`google_drive_folder_id`**: (Optional) ID of the Google Drive folder to upload to
3. **`upload_method`**: Choose "auto", "google_drive", or "imgbb"

**Example Configuration:**
- `google_drive_creds_path`: `C:\ComfyUI\credentials\google-drive-key.json`
- `google_drive_folder_id`: `1ABC123DEF456GHI789JKL` (optional)
- `upload_method`: `auto` (recommended)

With these settings, large images will automatically upload to your specified Google Drive folder, bypassing the 32MB ImgBB limit!