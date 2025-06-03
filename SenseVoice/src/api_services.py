import asyncio
import threading
import time
import uuid
from collections import deque
from typing import List, Dict, Any, Optional
import os

import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from loguru import logger
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Configuration parameters
NUM_MODELS = 2  # Number of model instances
MAX_BATCH_SIZE = 32  # Maximum batch size
BATCH_TIMEOUT = 1  # Batch timeout in seconds
REQUEST_TIMEOUT = 30  # Request timeout in seconds
MODEL_DIR = "/home/niwang/models/SenseVoiceSmall"

emotion_map = {
        "😊": "开心",
        "😔": "悲伤",
        "😡": "愤怒", 
        "😰": "恐惧",
        "🤢": "厌恶",
        "😮": "惊讶"
    }

app = FastAPI(title="SenseVoice ASR API")

class TranscriptionResponse(BaseModel):
    text: str
    emotion: str
    file_name: str

class BatchRequest:
    def __init__(self, file_path: str, future: asyncio.Future):
        self.file_path = file_path
        self.future = future
        self.timestamp = time.time()

class ModelInstance:
    def __init__(self, model_id: int):
        self.model_id = model_id
        self.model = AutoModel(
            model=MODEL_DIR,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            hub="hf",
            disable_update=True
        )
        self.batch_queue = deque()
        self.processing = False
        self.lock = threading.Lock()
        
    async def process_batch(self):
        while True:
            batch = []
            batch_files = []
            
            # Wait for requests or timeout
            while len(batch) < MAX_BATCH_SIZE:
                if not self.batch_queue:
                    if batch:
                        break
                    await asyncio.sleep(0.1)
                    continue
                
                request = self.batch_queue[0]
                if batch and time.time() - request.timestamp > BATCH_TIMEOUT:
                    break
                    
                request = self.batch_queue.popleft()
                batch.append(request)
                batch_files.append(request.file_path)
                
                if len(batch) >= MAX_BATCH_SIZE:
                    break
            
            if not batch:
                continue
                
            try:
                # Process batch
                results = self.model.generate(
                    input=batch_files,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=32,
                    merge_vad=True,
                    merge_length_s=15,
                )
                
                # Send results back
                for i, request in enumerate(batch):
                    try:
                        original_text = results[i]["text"]
                        text = rich_transcription_postprocess(original_text)
                        emoji = text[-1]
                        emotion = emotion_map.get(emoji, "中性")
                        file_name = os.path.basename(request.file_path)
                        
                        response = TranscriptionResponse(
                            text=text,
                            emotion=emotion,
                            file_name=file_name
                        )
                        request.future.set_result(response)
                        
                        # Cleanup temp file
                        try:
                            os.remove(request.file_path)
                        except:
                            pass
                            
                    except Exception as e:
                        request.future.set_exception(e)
                        
            except Exception as e:
                # If batch processing fails, fail all requests
                for request in batch:
                    request.future.set_exception(e)

class LoadBalancer:
    def __init__(self):
        self.models = [ModelInstance(i) for i in range(NUM_MODELS)]
        self.current_model = 0
        
    def get_next_model(self) -> ModelInstance:
        model = self.models[self.current_model]
        self.current_model = (self.current_model + 1) % len(self.models)
        return model

balancer = LoadBalancer()

@app.on_event("startup")
async def startup_event():
    # Create temp directory
    os.makedirs("tmp_audio", exist_ok=True)
    
    # Start batch processing for each model
    for model in balancer.models:
        asyncio.create_task(model.process_batch())
    
    logger.info(f"Started {NUM_MODELS} model instances")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Save uploaded file
        timestamp = int(time.time() * 1000)
        temp_file = f"tmp_audio/{timestamp}_{file.filename}"
        
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Create future for async result
        future = asyncio.get_running_loop().create_future()
        
        # Create batch request
        request = BatchRequest(temp_file, future)
        
        # Add to next available model's queue
        model = balancer.get_next_model()
        model.batch_queue.append(request)
        
        # Wait for result with timeout
        try:
            return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            # Clean up the temporary file if timeout occurs
            try:
                os.remove(temp_file)
            except:
                pass
            raise HTTPException(
                status_code=408,
                detail=f"Request processing timed out after {REQUEST_TIMEOUT} seconds"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
