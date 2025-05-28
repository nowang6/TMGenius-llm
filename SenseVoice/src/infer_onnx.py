# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import numpy as np
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
from ais_bench.infer.interface import InferSession
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank


class SenseVoiceOnnxModel():
    def __init__(self):
        super().__init__()
        self.blank_id = 0
        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.textnorm_dict = {'withitn': 14, "woitn": 15}

    def infer_onnx(
        self,
        data_in,
        om_sess,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        key = ["wav_file_tmp_name"]
        use_itn = kwargs.get('use_itn', False)
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )
        speech, speech_lengths = extract_fbank(
            audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
        )
        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language = torch.LongTensor([self.lid_dict[language] if language in self.lid_dict else 0]).to(speech.device)

        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm = torch.LongTensor([self.textnorm_dict[textnorm]]).to(speech.device)

        s = time.time()
        feed = [speech.cpu().detach().numpy().astype(np.float32),
                speech_lengths.cpu().detach().numpy().astype(np.int32),
                language.cpu().detach().numpy().astype(np.int32),
                textnorm.cpu().detach().numpy().astype(np.int32)]

        ctc_logits, encoder_out_lens = om_sess.infer(feed, mode='dymshape', custom_sizes=10000000)
        ctc_logits = torch.from_numpy(ctc_logits).npu()
        encoder_out_lens = torch.from_numpy(encoder_out_lens).npu()
        e = time.time()
        cost_time = e - s
        
        results = []

        x = ctc_logits[0, : encoder_out_lens[0].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)        
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()

        # Change integer-ids to tokens
        text = tokenizer.decode(token_int)
        result_i = {"key": key[0], "text": text}
        results.append(result_i)

        return results, cost_time

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="Sensevoice infer")
    parser.add_argument("--model_path", type=str, help="modelpath")
    parser.add_argument('--om_path', type=str, help='om model')
    parser.add_argument('--device', type=int, help='npu device num')
    parser.add_argument('--input', type=str, help='input audio file')
    parser.add_argument('--perform', type=bool, help='test performance')
    parser.add_argument('--loop', default=10, type=int, help='loop time')
    args = parser.parse_args()

    # 初始化pytorch模型

    _, kwargs = AutoModel.build_model(model=args.model_path, trust_remote_code=True)
    m = SenseVoiceOnnxModel()

    # 载入om模型
    om_sess = InferSession(args.device, args.om_path)

    with torch.no_grad():
        # 执行推理
        res, _ = m.infer_onnx(
            data_in=args.input,
            om_sess=om_sess,
            language="auto",
            use_itn=False,
            ban_emo_unk=False,
            **kwargs,
        )
        text = rich_transcription_postprocess(res[0]['text'])
        print('语音输出:')
        print(text)

        if args.perform:
            # 执行性能测试
            t = 0
            for _ in range(args.loop):
                res, cost_time = m.infer_onnx(
                    data_in=args.input,
                    om_sess=om_sess,
                    language="auto",
                    use_itn=False,
                    ban_emo_unk=False,
                    **kwargs,
                )
                t += cost_time
            print('单条数据推理耗时：')
            print(str(t / args.loop))     

