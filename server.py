from aiohttp import web
from werkzeug.utils import secure_filename
import os
import folder_paths

@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

async def upload_audio(request):
    try:
        data = await request.post()
        file = data['files[]']  # Match the JS formData key
        
        if not file:
            return web.json_response({"error": "No file provided"}, status=400)
            
        # Save to input directory
        filename = secure_filename(file.filename)
        save_path = os.path.join(folder_paths.get_input_directory(), filename)
        
        # Ensure the file has a valid audio extension
        valid_extensions = ['.wav', '.mp3', '.ogg', '.m4a', '.flac']
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            return web.json_response({"error": "Invalid audio file type"}, status=400)
            
        # Save the file
        with open(save_path, 'wb') as f:
            f.write(file.file.read())
            
        return web.json_response({"path": save_path})
        
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return web.json_response({"error": str(e)}, status=500)

# Add routes to the server
def setup_routes(app):
    app.router.add_post("/upload/audio", upload_audio)
    app.middlewares.append(cors_middleware) 

@PromptServer.instance.routes.post("/memo/validate_audio")
async def validate_audio(request):
    try:
        data = await request.json()
        file_path = data.get("path", "")
        
        if not file_path:
            return web.json_response({"valid": False, "error": "No path provided"})
            
        # Check if path exists and is a valid audio file
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.wav', '.mp3', '.ogg', '.m4a', '.flac']:
                return web.json_response({"valid": True, "path": file_path})
                
        return web.json_response({"valid": False, "error": "Invalid audio file"})
        
    except Exception as e:
        return web.json_response({"valid": False, "error": str(e)})