from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

app = FastAPI()

model_dir = "/home/niwang/models/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

class TranscriptionResponse(BaseModel):
    text: str
    emotion: str
    raw_file: List[Dict[str, Any]]

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = "auto",
    use_itn: Optional[bool] = True
):
    try:
        # 保存上传的文件到临时目录
        temp_file = f"/tmp/{file.filename}"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        res = model.generate(
            input=temp_file,
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        
        # 删除临时文件
        os.remove(temp_file)
        
        text = rich_transcription_postprocess(res[0]["text"])
        emotion = text[-1]
        if emotion == "😊":
            emotion = "开心"
        elif emotion == "😔":
            emotion = "悲伤" 
        elif emotion == "😡":
            emotion = "愤怒"
        elif emotion == "😰":
            emotion = "恐惧"
        elif emotion == "🤢":
            emotion = "厌恶"
        elif emotion == "😮":
            emotion = "惊讶"
        else:
            emotion = "中立"
        
        return {
            "text": text,
            "emotion": emotion,
            "raw_file": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


