# Google Drive Setup for ComfyUI Upsampler

**⚠️ IMPORTANT NOTICE**: Google has changed their policy - service accounts can no longer upload to personal Google Drive accounts due to storage quota limitations. This setup only works with Google Workspace accounts that have Shared Drives. **For most users, ImgBB is recommended instead.**

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
4. **IMPORTANT**: Select "JSON" format (not P12)
5. Click "Create" - this will download a JSON file
6. Save it securely (e.g., `C:\ComfyUI\credentials\google-drive-creds.json`)

**What the downloaded file should look like:**
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "service-account-name@project-id.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}
```

**⚠️ IMPORTANT**: Make sure you download a "Service Account" JSON file, not an "OAuth client" or "API key" file!

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
1. **Create or navigate to the desired folder** in Google Drive
2. **Share the folder with your service account**:
   - Right-click the folder → "Share"
   - Add your service account email (from your JSON file's `client_email` field)
   - Example: `upsampler-bot@comfyui-upsampler.iam.gserviceaccount.com`
   - Set permission to "Editor"
   - Click "Send"
3. **Copy the folder ID** from the URL:
   - Example URL: `https://drive.google.com/drive/folders/1ABC123DEF456GHI789JKL`
   - Folder ID: `1ABC123DEF456GHI789JKL`
4. **Paste this ID** into the `google_drive_folder_id` field in the node

**⚠️ IMPORTANT**: The service account must have access to the folder, or you'll get a "File not found" error!

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

### "Service account info was not in the expected format"
This error means you're using the wrong type of credentials file:

**Problem**: You downloaded an OAuth client credentials or API key file instead of a service account file.

**Solution**:
1. Go back to Google Cloud Console
2. Navigate to "IAM & Admin" > "Service Accounts" (not "Credentials")
3. Click on your service account name
4. Go to "Keys" tab
5. Create a new JSON key
6. Make sure the downloaded file has `"type": "service_account"` in it

**Wrong file types to avoid**:
- OAuth 2.0 client credentials (`"type": "web"` or `"type": "installed"`)
- API keys (just a string, not JSON)
- Application default credentials

## Quick Setup Summary

Once you've completed the setup, your node will have these new fields:

1. **`google_drive_creds_path`**: Path to your downloaded JSON credentials file
2. **`google_drive_folder_id`**: (Optional) ID of the Google Drive folder to upload to
3. **`upload_method`**: Choose "auto", "google_drive", or "imgbb"

**Example Configuration:**
- `google_drive_creds_path`: `C:\ComfyUI\credentials\google-drive-key.json`
- `google_drive_folder_id`: `1ABC123DEF456GHI789JKL` (optional)
- `upload_method`: `auto` (recommended)

**Important:** Make sure to provide the FULL PATH to the JSON file, not just the directory!

With these settings, large images will automatically upload to your specified Google Drive folder, bypassing the 32MB ImgBB limit!