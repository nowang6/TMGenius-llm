from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os


model_dir = "/home/niwang/models/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    hub="hf",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
emotion = res[0].get("emotion", "unknown")  # 获取情感识别结果

print(res)
print(f"识别文本: {text}")
print(f"情感状态: {emotion}")

