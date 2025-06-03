from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os

model_dir = "/home/niwang/models/SenseVoiceSmall"

# 初始化模型
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

# 准备多个音频文件路径
audio_files = [
    "data/悲伤.mp3",
    "data/悲伤2.mp3",
    "data/悲伤3.mp3"
    # 添加更多音频文件路径
    # "/path/to/audio2.mp3",
    # "/path/to/audio3.mp3",
]

# 批量推理
results = model.generate(
    input=audio_files,  # 直接传入文件路径列表
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)

# 处理每个音频的识别结果
for i, res in enumerate(results):
    text = rich_transcription_postprocess(res["text"])
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
    print(f"\n音频 {i+1}:")
    print(f"识别文本: {text}")
    print(f"情感状态: {emotion}")
