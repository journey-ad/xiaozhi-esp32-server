import time
import wave
import os
import sys
import io
from config.logger import setup_logging
from typing import Optional, Tuple, List, Union
import uuid
import opuslib_next
import torch
import librosa
import numpy as np

from core.providers.asr.base import ASRProviderBase

from modelscope import snapshot_download
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TAG = __name__
logger = setup_logging()


# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class SenseVoiceSmall:
    """
    @description SenseVoiceSmall模型类，负责语音识别
    @param {str} model_dir - 模型目录
    @param {int} batch_size - 批处理大小
    @param {Union[str, int]} device_id - 设备ID
    @param {str} plot_timestamp_to - 时间戳绘制目标
    @param {bool} quantize - 是否量化
    @param {int} intra_op_num_threads - 线程数
    @param {str} cache_dir - 缓存目录
    """
    def __init__(self, model_dir: str = None, batch_size: int = 1,
                 punc_model_dir: str = None,
                 device_id: str = "-1", plot_timestamp_to: str = "",
                 intra_op_num_threads: int = 4, cache_dir: str = None,
                 hotwords: str = "", hotwords_score: float = 1.0, **kwargs):

        # 下载模型
        model_dir = self._download_model(model_dir, cache_dir)

        sys.path.append(model_dir)

        # 导入相关模块
        self._import_modules()

        model_file = os.path.join(model_dir, "model.onnx")

        self._initialize_model(model_dir=model_dir, model_file=model_file, punc_model_dir=punc_model_dir, device_id=device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = batch_size
        self.blank_id = 0

        self.enable_hotwords = False
        if hotwords:
            # 预生成热词
            logger.bind(tag=TAG).info(f"预编译ASR热词...")
            self.hotwords_list, self.hotwords_length = self.proc_hotword(hotwords)
            self.enable_hotwords = True
            self.context_emb = self.ort_infer_emb([self.hotwords_list, self.hotwords_length])[0]
            self.hotwords_score = np.array([hotwords_score], dtype=np.float32)
            logger.bind(tag=TAG).info(f"共处理{len(self.hotwords_list)}个热词: {hotwords[:100]}{'...' if len(hotwords) > 100 else ''}")

    def _import_modules(self):
        """导入所需的模块"""
        global OrtInferSession, read_yaml, SentencepiecesTokenizer, WavFrontend, pad_list
        from funasr_onnx.utils.utils import OrtInferSession, read_yaml
        from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
        from funasr_onnx.utils.frontend import WavFrontend
        from funasr_onnx.utils.utils import pad_list

    def _download_model(self, model_dir: str, cache_dir: str) -> str:
        """下载模型"""
        try:
            logger.bind(tag=TAG).info(f"检查模型文件...")
            from modelscope.hub.snapshot_download import snapshot_download
            return snapshot_download(repo_id="dengcunqin/SenseVoiceSmall_hotword", local_dir=model_dir, cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {e}")

    def _initialize_model(self, model_dir: str, model_file: str, punc_model_dir: str, device_id: Union[str, int], intra_op_num_threads: int) -> None:
        """初始化模型"""
        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
        config["frontend_conf"]["cmvn_file"] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_emb = OrtInferSession(os.path.join(model_dir, "sensevoice_model_hot_emb.onnx"), device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_hot_module = OrtInferSession(os.path.join(model_dir, "sensevoice_model_hot_module.onnx"), device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer_nohot_module = OrtInferSession(os.path.join(model_dir, "sensevoice_model_nohot_module.onnx"), device_id, intra_op_num_threads=intra_op_num_threads)

        # 标点恢复
        self.punc_model = None
        if punc_model_dir:
            self.punc_model = AutoModel(model="ct-punc", model_dir=punc_model_dir, model_revision="v2.0.4")

    def proc_hotword(self, hotwords: str) -> Tuple[np.ndarray, np.ndarray]:
        """处理热词"""
        hotword_str_list = hotwords.split("|")
        hotword_list = [np.array(self.tokenizer.encode(i), dtype=np.int64) for i in hotword_str_list] if hotword_str_list else [np.array([1], dtype=np.int64)]
        hotword_list.insert(0, np.array([1], dtype=np.int64))
        hotwords_length = np.array([len(i) for i in hotword_list], dtype=np.int32)
        max_length = max(len(arr) for arr in hotword_list)
        hotwords = pad_list(hotword_list, pad_value=-1, max_len=max_length)
        return hotwords.astype(np.int64), hotwords_length

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs) -> List[str]:
        """调用模型进行语音识别"""

        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        asr_res = []

        for beg_idx in range(0, len(waveform_list), self.batch_size):
            end_idx = min(len(waveform_list), beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            encoder_out, encoder_out_lens = self.infer(feats, feats_len)

            ctc_logits = self.ort_infer_hot_module([encoder_out, self.context_emb, self.hotwords_score])[0] if self.enable_hotwords else self.ort_infer_nohot_module([encoder_out])[0]

            for b in range(feats.shape[0]):
                # back to torch.Tensor
                if isinstance(ctc_logits, np.ndarray):
                    ctc_logits = torch.from_numpy(ctc_logits).float()
                # support batch_size=1 only currently
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)
                token_int = yseq[yseq != self.blank_id].tolist()
                asr_res.append(self.tokenizer.decode(token_int))

        for i, res in enumerate(asr_res):
            res = rich_transcription_postprocess(res)
            if self.punc_model:
                res = self.punc_model.generate(res)[0]["text"]
            asr_res[i] = res

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List[np.ndarray]:
        """加载音频数据"""
        def load_wav(path: str) -> np.ndarray:
            return librosa.load(path, sr=fs)[0]

        if isinstance(wav_content, np.ndarray):
            return [wav_content]
        if isinstance(wav_content, str):
            return [load_wav(wav_content)]
        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征"""
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        """填充特征"""
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        return np.array([pad_feat(feat, feat.shape[0]) for feat in feats]).astype(np.float32)

    def infer(self, feats: np.ndarray, feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """推理"""
        return self.ort_infer([feats, feats_len])


class ASRProvider(ASRProviderBase):
    """
    @description ASR提供者类，负责语音识别的整体流程
    """
    def __init__(self, config: dict, delete_audio_file: bool):
        """
        @param {dict} config - 配置信息
        @param {bool} delete_audio_file - 是否删除临时音频文件
        """
        self.model_dir = config.get("model_dir", "models/SenseVoiceSmall_hotword")
        self.punc_model_dir = config.get("punc_model_dir", "models/SenseVoiceSmall_hotword/ct-punc")
        self.output_dir = config.get("output_dir")
        self.hotwords = config.get("hotwords", "")
        self.hotwords_str = ""
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

        with CaptureOutput():
            logger.bind(tag=TAG).info(f"FunASR_hotword 初始化...")

            # hotwords
            if self.hotwords.strip():
                # 如果热词配置是文件路径
                if os.path.exists(self.hotwords):
                    with open(self.hotwords, encoding="utf-8") as f:
                        hotwords_list = []
                        for line in f.readlines():
                            line = line.strip()
                            # 跳过空行和注释行
                            if line and not line.startswith("#"):
                                # 格式：word:score 或者 word
                                # 仅提取word部分
                                word = line.split("#")[0].split(":")[0].strip()
                                hotwords_list.append(word)

                        self.hotwords_str = "|".join(hotwords_list)
                else:
                    self.hotwords_str = self.hotwords

            self.model = SenseVoiceSmall(
                model_dir=self.model_dir,
                punc_model_dir=self.punc_model_dir,
                batch_size=10,
                hotwords=self.hotwords_str, 
                hotwords_score=1.5
            )

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    async def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 语音识别
            start_time = time.time()
            result = self.model(file_path, hotwords_str=self.hotwords_str, hotwords_score=1.0)
            text = "".join(result)
            logger.bind(tag=TAG).debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return "", None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"文件删除失败: {file_path} | 错误: {e}")
