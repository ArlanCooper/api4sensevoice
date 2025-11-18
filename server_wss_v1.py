"""
WebSocket语音识别服务器
基于FastAPI和FunASR实现实时语音识别服务
"""

# ==================== 导入依赖库 ====================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
from urllib.parse import parse_qs
from typing import Optional
from loguru import logger
import numpy as np
import argparse
import uvicorn
import sys
import json
import traceback
import time


# ==================== 日志配置 ====================
def setup_logging():
    """配置日志系统"""
    logger.remove()
    
    log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
    
    # 标准输出：DEBUG、INFO、WARNING级别
    logger.add(
        sys.stdout,
        format=log_format,
        level="DEBUG",
        filter=lambda record: record["level"].no < 40
    )
    
    # 标准错误：ERROR、CRITICAL级别
    logger.add(
        sys.stderr,
        format=log_format,
        level="ERROR",
        filter=lambda record: record["level"].no >= 40
    )


setup_logging()


# ==================== 配置类定义 ====================
class Config(BaseSettings):
    """应用配置类，支持从环境变量读取配置"""
    
    chunk_size_ms: int = Field(300, description="音频块大小（毫秒）")
    sample_rate: int = Field(16000, description="采样率（Hz）")
    bit_depth: int = Field(16, description="位深度")
    channels: int = Field(1, description="声道数")
    avg_logprob_thr: float = Field(-0.25, description="平均对数概率阈值")


config = Config()


# ==================== 表情符号映射配置 ====================
# 情感标签映射
EMO_DICT = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

# 事件标签映射
EVENT_DICT = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

# 完整标签映射
EMOJI_DICT = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

# 语言标签统一替换
LANG_DICT = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

# 表情符号集合
EMO_SET = {"😊", "😔", "😡", "😰", "🤢", "😮"}
EVENT_SET = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


# ==================== 文本格式化函数 ====================
def format_str_v2(text: str) -> str:
    """
    格式化识别文本：统计标签出现次数，选择主要情感和事件
    
    Args:
        text: 原始识别文本（包含ASR模型输出的各种标签）
    
    Returns:
        格式化后的文本（事件表情在前，文本在中，情感表情在后）
    """
    sptk_dict = {}
    
    # 统计标签出现次数并移除标签
    for sptk in EMOJI_DICT:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")
    
    # 选择主要情感（出现次数最多的）
    emo = "<|NEUTRAL|>"
    for e in EMO_DICT:
        if sptk_dict.get(e, 0) > sptk_dict.get(emo, 0):
            emo = e
    
    # 添加事件表情到开头
    for e in EVENT_DICT:
        if sptk_dict.get(e, 0) > 0:
            text = EVENT_DICT[e] + text
            break
    
    # 添加情感表情到末尾
    text = text + EMO_DICT[emo]
    
    # 移除表情符号前后的空格
    for emoji in EMO_SET.union(EVENT_SET):
        text = text.replace(" " + emoji, emoji)
        text = text.replace(emoji + " ", emoji)
    
    return text.strip()


def format_str_v3(text: str) -> str:
    """
    格式化识别文本：处理多语言分段，合并相同情感和事件
    
    Args:
        text: 原始识别文本（可能包含多语言分段和多个标签）
    
    Returns:
        格式化后的统一文本（合并了相同的情感/事件，去除了重复标记）
    """
    def get_emo(s: str) -> Optional[str]:
        """获取文本末尾的情感表情符号"""
        return s[-1] if s and s[-1] in EMO_SET else None
    
    def get_event(s: str) -> Optional[str]:
        """获取文本开头的事件表情符号"""
        return s[0] if s and s[0] in EVENT_SET else None
    
    # 处理特殊情况
    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")
    
    # 统一语言标签
    for lang in LANG_DICT:
        text = text.replace(lang, "<|lang|>")
    
    # 按语言分段格式化
    s_list = [format_str_v2(s_i).strip(" ") for s_i in text.split("<|lang|>")]
    
    if not s_list:
        return ""
    
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    
    # 合并相同的事件和情感标记
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        
        # 合并相同的事件标记
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) is not None:
            s_list[i] = s_list[i][1:]
        
        cur_ent_event = get_event(s_list[i])
        
        # 合并相同的情感标记
        if get_emo(s_list[i]) is not None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        
        new_s += s_list[i].strip().lstrip()
    
    # 修正特定错误模式
    new_s = new_s.replace("The.", " ")
    
    return new_s.strip()


# ==================== 模型初始化 ====================
def init_models():
    """初始化ASR和VAD模型"""
    logger.info("正在加载ASR模型...")
    model_asr = AutoModel(
        model="/home/octopus/data/llm_list/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
        disable_update=True
    )
    logger.info("ASR模型加载完成")
    
    logger.info("正在加载VAD模型...")
    model_vad = AutoModel(
        model="/home/octopus/data/llm_list/fsmn_vad",
        model_revision="v2.0.4",
        disable_pbar=True,
        max_end_silence_time=500,
        disable_update=True
    )
    logger.info("VAD模型加载完成")
    
    return model_asr, model_vad


model_asr, model_vad = init_models()


# ==================== ASR处理函数 ====================
def asr_process(audio: np.ndarray, lang: str, cache: dict, use_itn: bool = False):
    """
    自动语音识别处理
    
    Args:
        audio: 音频数据（numpy数组，float32格式，范围[-1.0, 1.0]）
        lang: 语言代码（如"zh"中文、"en"英文、"auto"自动检测）
        cache: ASR缓存字典（用于流式识别，保持上下文状态）
        use_itn: 是否使用逆文本规范化，默认False
    
    Returns:
        识别结果列表，每个元素包含识别文本和元数据
    """
    start_time = time.time()
    
    result = model_asr.generate(
        input=audio,
        cache=cache,
        language=lang.strip(),
        use_itn=use_itn,
        batch_size_s=60,
    )
    
    elapsed_time = (time.time() - start_time) * 1000
    logger.debug(f"ASR处理耗时: {elapsed_time:.2f} 毫秒")
    
    return result


# ==================== FastAPI应用初始化 ====================
app = FastAPI(title="WebSocket语音识别服务")

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 异常处理 ====================
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error("发生异常", exc_info=True)
    
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = f"验证错误: {str(exc.errors())}"
    else:
        status_code = 500
        message = f"内部服务器错误: {str(exc)}"
    
    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            info=message,
            data=""
        ).model_dump()
    )


# ==================== 响应模型定义 ====================
class TranscriptionResponse(BaseModel):
    """转录响应数据模型"""
    code: int  # 状态码：0=成功，2=检测到语音，400=客户端错误，500=服务器错误
    info: str  # 信息字段：通常包含JSON格式的详细识别结果或错误信息
    data: str  # 数据字段：格式化后的识别文本


# ==================== WebSocket端点 ====================
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket语音识别端点"""
    # 初始化缓冲区
    audio_buffer = np.array([], dtype=np.float32)
    audio_vad = np.array([], dtype=np.float32)
    cache_vad = {}
    cache_asr = {}
    
    try:
        # 解析查询参数
        query_params = parse_qs(websocket.scope['query_string'].decode())
        lang = query_params.get('lang', ['auto'])[0].lower()
        
        # 接受连接
        await websocket.accept()
        logger.info(f"WebSocket连接已建立，语言设置: {lang}")
        
        # 计算音频块大小
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        if chunk_size <= 0:
            error_response = TranscriptionResponse(
                code=500,
                info="无效的块大小配置",
                data=""
            )
            await websocket.send_json(error_response.model_dump())
            await websocket.close()
            return
        
        # 初始化状态变量
        last_vad_beg = last_vad_end = -1
        offset = 0
        speech_detected_sent = False
        buffer = b""
        
        # 主处理循环
        while True:
            try:
                data = await websocket.receive_bytes()
            except Exception as e:
                logger.error(f"接收数据错误: {e}")
                error_response = TranscriptionResponse(
                    code=500,
                    info=f"接收音频数据错误: {str(e)}",
                    data=""
                )
                try:
                    await websocket.send_json(error_response.model_dump())
                except:
                    pass
                break
            
            # 追加到字节缓冲区
            buffer += data
            
            if len(buffer) < 2:
                continue
            
            # 转换为音频数组
            try:
                audio_data = np.frombuffer(
                    buffer[:len(buffer) - (len(buffer) % 2)],
                    dtype=np.int16
                ).astype(np.float32) / 32767.0
                audio_buffer = np.append(audio_buffer, audio_data)
            except Exception as e:
                logger.error(f"处理音频缓冲区错误: {e}")
                error_response = TranscriptionResponse(
                    code=500,
                    info=f"处理音频数据错误: {str(e)}",
                    data=""
                )
                try:
                    await websocket.send_json(error_response.model_dump())
                except:
                    pass
                continue
            
            # 保留未处理的字节
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
            
            # 处理音频块
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)
                
                # 发送语音检测消息
                if last_vad_beg > 1 and not speech_detected_sent:
                    response = TranscriptionResponse(
                        code=2,
                        info="detect speech",
                        data=''
                    )
                    await websocket.send_json(response.model_dump())
                    speech_detected_sent = True
                
                # VAD处理
                try:
                    vad_result = model_vad.generate(
                        input=chunk,
                        cache=cache_vad,
                        is_final=False,
                        chunk_size=config.chunk_size_ms
                    )
                except Exception as e:
                    logger.error(f"VAD处理错误: {e}")
                    error_response = TranscriptionResponse(
                        code=500,
                        info=f"VAD处理错误: {str(e)}",
                        data=""
                    )
                    try:
                        await websocket.send_json(error_response.model_dump())
                    except:
                        pass
                    continue
                
                # 处理VAD结果
                if len(vad_result) > 0 and len(vad_result[0]["value"]):
                    vad_segments = vad_result[0]["value"]
                    
                    for segment in vad_segments:
                        if segment[0] > -1:
                            last_vad_beg = segment[0]
                        if segment[1] > -1:
                            last_vad_end = segment[1]
                        
                        # 检测到完整语音段
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            
                            # 转换为样本索引
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            
                            # 边界检查
                            if beg < 0 or end < beg or end > len(audio_vad):
                                logger.warning(
                                    f"无效的VAD段: beg={beg}, end={end}, "
                                    f"audio_vad_len={len(audio_vad)}"
                                )
                                audio_vad = (
                                    audio_vad[max(0, end):]
                                    if end < len(audio_vad)
                                    else np.array([], dtype=np.float32)
                                )
                                last_vad_beg = last_vad_end = -1
                                speech_detected_sent = False
                                continue
                            
                            logger.info(f"[VAD段] 音频长度: {end - beg} 样本")
                            
                            # ASR识别
                            try:
                                result = asr_process(
                                    audio_vad[beg:end],
                                    lang.strip(),
                                    cache_asr,
                                    use_itn=True
                                )
                                logger.info(f"ASR响应: {result}")
                            except Exception as e:
                                logger.error(f"ASR处理错误: {e}")
                                error_response = TranscriptionResponse(
                                    code=500,
                                    info=f"ASR处理错误: {str(e)}",
                                    data=""
                                )
                                try:
                                    await websocket.send_json(error_response.model_dump())
                                except:
                                    pass
                                result = None
                            
                            # 清理已处理的音频
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            speech_detected_sent = False
                            
                            # 发送识别结果
                            if result is not None:
                                try:
                                    formatted_text = (
                                        format_str_v3(result[0]['text'])
                                        if result and len(result) > 0 and 'text' in result[0]
                                        else ""
                                    )
                                    
                                    response = TranscriptionResponse(
                                        code=0,
                                        info=(
                                            json.dumps(result[0], ensure_ascii=False)
                                            if result and len(result) > 0
                                            else ""
                                        ),
                                        data=formatted_text
                                    )
                                    await websocket.send_json(response.model_dump())
                                except Exception as e:
                                    logger.error(f"格式化或发送ASR结果错误: {e}")
                                    error_response = TranscriptionResponse(
                                        code=500,
                                        info=f"格式化结果错误: {str(e)}",
                                        data=""
                                    )
                                    try:
                                        await websocket.send_json(error_response.model_dump())
                                    except:
                                        pass
    
    except WebSocketDisconnect:
        logger.info("WebSocket连接已断开")
    except Exception as e:
        logger.error(f"WebSocket端点意外错误: {e}\n调用堆栈:\n{traceback.format_exc()}")
        try:
            error_response = TranscriptionResponse(
                code=500,
                info=f"内部服务器错误: {str(e)}",
                data=""
            )
            await websocket.send_json(error_response.model_dump())
        except:
            pass
        try:
            await websocket.close()
        except:
            pass
    finally:
        # 清理资源
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache_vad.clear()
        cache_asr.clear()
        logger.info("WebSocket断开后资源已清理")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行FastAPI语音识别服务")
    parser.add_argument(
        '--port',
        type=int,
        default=8034,
        help='服务端口号'
    )
    
    args = parser.parse_args()
    logger.info(f"启动服务，端口: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

