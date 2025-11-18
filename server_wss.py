# ==================== 导入依赖库 ====================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
# FastAPI: 现代、快速的Web框架，用于构建API服务
# WebSocket: WebSocket连接处理类，用于实时双向通信
# WebSocketDisconnect: WebSocket断开连接的异常类
# Request: HTTP请求对象，用于获取请求信息
# HTTPException: HTTP异常类，用于抛出HTTP错误

from fastapi.exceptions import RequestValidationError
# RequestValidationError: 请求验证错误异常，当请求数据不符合模型定义时抛出

from fastapi.responses import JSONResponse
# JSONResponse: JSON格式的HTTP响应类，用于返回JSON数据

from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware: 跨域资源共享中间件，允许浏览器跨域访问API

from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
# HTTP_422_UNPROCESSABLE_ENTITY: HTTP 422状态码常量，表示请求格式正确但语义错误

from pydantic_settings import BaseSettings
# BaseSettings: Pydantic配置基类，用于管理应用配置，支持从环境变量读取

from pydantic import BaseModel, Field
# BaseModel: Pydantic数据模型基类，用于数据验证和序列化
# Field: 字段验证和描述工具，用于定义模型字段的约束和说明

from funasr import AutoModel
# AutoModel: FunASR自动语音识别模型加载器，用于加载ASR和VAD模型

import numpy as np
# numpy: 数值计算库，用于音频数据的数组操作和数学运算

import soundfile as sf
# soundfile: 音频文件读写库，用于读取WAV等音频文件

import argparse
# argparse: 命令行参数解析库，用于解析启动参数

import uvicorn
# uvicorn: ASGI服务器，用于运行FastAPI应用

from urllib.parse import parse_qs
# parse_qs: URL查询参数解析函数，用于解析WebSocket连接URL中的查询参数

import os
# os: 操作系统接口库，用于文件路径操作和系统调用

from modelscope.pipelines import pipeline
# pipeline: ModelScope模型管道，用于加载和使用预训练模型（说话人验证）

from loguru import logger
# logger: Loguru日志库，提供结构化、高性能的日志记录功能

import sys
# sys: 系统相关参数和函数，用于访问标准输入输出流

import json
# json: JSON数据处理库，用于JSON序列化和反序列化

import traceback
# traceback: 异常堆栈跟踪库，用于获取详细的异常信息

import time
# time: 时间相关函数库，用于时间戳和性能统计

# ==================== 日志配置 ====================
logger.remove()
# 移除Loguru的默认日志处理器，以便自定义日志输出

log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
# 定义日志格式字符串：
# {time:YYYY-MM-DD HH:mm:ss}: 时间戳，格式为年-月-日 时:分:秒
# [{level}]: 日志级别（DEBUG、INFO、WARNING、ERROR等）
# {file}:{line}: 文件名和行号
# {message}: 日志消息内容

logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
# 添加标准输出日志处理器：
# sys.stdout: 输出到标准输出（控制台）
# format: 使用上面定义的日志格式
# level="DEBUG": 设置日志级别为DEBUG（最低级别，会记录所有日志）
# filter: 过滤函数，只输出日志级别编号小于40的日志（DEBUG、INFO、WARNING级别）
# 日志级别编号：DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50

logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)
# 添加标准错误日志处理器：
# sys.stderr: 输出到标准错误流（控制台错误输出）
# format: 使用上面定义的日志格式
# level="ERROR": 设置日志级别为ERROR
# filter: 过滤函数，只输出日志级别编号大于等于40的日志（ERROR、CRITICAL级别）
# 这样可以将错误日志单独输出到stderr，便于区分和重定向


# ==================== 配置类定义 ====================
class Config(BaseSettings):
    # 应用配置类，继承自Pydantic的BaseSettings
    # BaseSettings支持从环境变量、.env文件等自动读取配置
    # 这些配置参数可以通过环境变量覆盖默认值
    
    sv_thr: float = Field(0.3, description="Speaker verification threshold")
    # 说话人验证阈值（浮点数类型）
    # 默认值0.3：当说话人验证相似度分数 >= 0.3时，认为匹配成功
    # 范围通常在0.0-1.0之间，值越大要求越严格
    
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    # 音频块大小（整数类型，单位：毫秒）
    # 默认值300ms：每次处理300毫秒的音频数据
    # 用于将连续音频流分割成固定大小的块进行处理
    
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    # 采样率（整数类型，单位：Hz）
    # 默认值16000：16kHz采样率，这是语音识别常用的采样率
    # 表示每秒采样16000个点，决定了音频的频率范围（最高8000Hz）
    
    bit_depth: int = Field(16, description="Bit depth")
    # 位深度（整数类型，单位：位）
    # 默认值16：16位音频，每个采样点用16位（2字节）表示
    # 决定了音频的动态范围，16位可表示65536个不同的值
    
    channels: int = Field(1, description="Number of audio channels")
    # 声道数（整数类型）
    # 默认值1：单声道（mono），只有一个音频通道
    # 如果是立体声则为2，多声道音频可能更多
    
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")
    # 平均对数概率阈值（浮点数类型）
    # 默认值-0.25：用于过滤低质量的ASR识别结果
    # 如果识别结果的平均对数概率低于此阈值，可能被认为是低质量结果

config = Config()
# 创建配置实例，自动从环境变量或默认值加载配置

# ==================== 表情符号映射字典 ====================
emo_dict = {
    # 情感标签到表情符号的映射字典
    # 用于将ASR模型输出的情感标签转换为可读的表情符号
	"<|HAPPY|>": "😊",      # 开心/高兴：映射到笑脸表情
	"<|SAD|>": "😔",        # 悲伤：映射到沮丧表情
	"<|ANGRY|>": "😡",      # 愤怒：映射到愤怒表情
	"<|NEUTRAL|>": "",      # 中性：无表情符号（空字符串）
	"<|FEARFUL|>": "😰",    # 恐惧：映射到恐惧表情
	"<|DISGUSTED|>": "🤢",  # 厌恶：映射到恶心表情
	"<|SURPRISED|>": "😮",  # 惊讶：映射到惊讶表情
}

event_dict = {
    # 事件标签到表情符号的映射字典
    # 用于将ASR模型识别出的音频事件转换为表情符号
	"<|BGM|>": "🎼",        # 背景音乐：映射到音乐符号
	"<|Speech|>": "",       # 语音：无表情符号（普通语音不需要标记）
	"<|Applause|>": "👏",   # 掌声：映射到鼓掌表情
	"<|Laughter|>": "😀",   # 笑声：映射到大笑表情
	"<|Cry|>": "😭",        # 哭声：映射到哭泣表情
	"<|Sneeze|>": "🤧",     # 喷嚏：映射到打喷嚏表情
	"<|Breath|>": "",       # 呼吸：无表情符号（通常不需要标记）
	"<|Cough|>": "🤧",      # 咳嗽：映射到打喷嚏表情（与喷嚏共用）
}

emoji_dict = {
    # 完整的标签到表情符号映射字典（包含所有可能的标签）
    # 这是最全面的映射表，包含语言、情感、事件等所有标签
	"<|nospeech|><|Event_UNK|>": "❓",  # 无语音且未知事件：映射到问号表情
	"<|zh|>": "",                       # 中文语言标签：无表情符号（语言标签通常不显示）
	"<|en|>": "",                       # 英文语言标签：无表情符号
	"<|yue|>": "",                      # 粤语语言标签：无表情符号
	"<|ja|>": "",                       # 日语语言标签：无表情符号
	"<|ko|>": "",                       # 韩语语言标签：无表情符号
	"<|nospeech|>": "",                 # 无语音标签：无表情符号
	"<|HAPPY|>": "😊",                  # 开心：映射到笑脸表情
	"<|SAD|>": "😔",                    # 悲伤：映射到沮丧表情
	"<|ANGRY|>": "😡",                  # 愤怒：映射到愤怒表情
	"<|NEUTRAL|>": "",                  # 中性：无表情符号
	"<|BGM|>": "🎼",                    # 背景音乐：映射到音乐符号
	"<|Speech|>": "",                   # 语音：无表情符号
	"<|Applause|>": "👏",               # 掌声：映射到鼓掌表情
	"<|Laughter|>": "😀",               # 笑声：映射到大笑表情
	"<|FEARFUL|>": "😰",                # 恐惧：映射到恐惧表情
	"<|DISGUSTED|>": "🤢",              # 厌恶：映射到恶心表情
	"<|SURPRISED|>": "😮",              # 惊讶：映射到惊讶表情
	"<|Cry|>": "😭",                    # 哭声：映射到哭泣表情
	"<|EMO_UNKNOWN|>": "",              # 未知情感：无表情符号
	"<|Sneeze|>": "🤧",                 # 喷嚏：映射到打喷嚏表情
	"<|Breath|>": "",                   # 呼吸：无表情符号
	"<|Cough|>": "😷",                  # 咳嗽：映射到口罩表情（与emo_dict中的不同）
	"<|Sing|>": "",                     # 唱歌：无表情符号
	"<|Speech_Noise|>": "",             # 语音噪声：无表情符号
	"<|withitn|>": "",                  # 内部标记（可能是模型内部使用的标记）：无表情符号
	"<|woitn|>": "",                    # 内部标记：无表情符号
	"<|GBG|>": "",                      # 内部标记：无表情符号
	"<|Event_UNK|>": "",                # 未知事件：无表情符号
}

lang_dict = {
    # 语言标签统一替换字典
    # 将所有不同的语言标签统一替换为"<|lang|>"，用于多语言文本的分段处理
    # 这样可以将不同语言的文本段用统一的标记分隔
    "<|zh|>": "<|lang|>",      # 中文标签替换为统一的语言标记
    "<|en|>": "<|lang|>",      # 英文标签替换为统一的语言标记
    "<|yue|>": "<|lang|>",     # 粤语标签替换为统一的语言标记
    "<|ja|>": "<|lang|>",      # 日语标签替换为统一的语言标记
    "<|ko|>": "<|lang|>",      # 韩语标签替换为统一的语言标记
    "<|nospeech|>": "<|lang|>", # 无语音标签也替换为统一的语言标记（用于分段）
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
# 情感表情符号集合（Python集合类型，用于快速查找）
# 包含所有可能出现在文本末尾的情感表情符号
# 用于format_str_v3函数中识别和合并相同的情感标记

event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}
# 事件表情符号集合（Python集合类型，用于快速查找）
# 包含所有可能出现在文本开头的事件表情符号
# 用于format_str_v3函数中识别和合并相同的事件标记

# ==================== 文本格式化函数 ====================
def format_str_v2(s):
    # 第二版格式化函数：统计标签出现次数，选择主要情感和事件
    # 参数: s - 原始识别文本（包含ASR模型输出的各种标签）
    # 返回: 格式化后的文本（事件表情在前，文本在中，情感表情在后）
    
	sptk_dict = {}
	# 初始化标签统计字典，用于记录每个标签在文本中出现的次数
	
	for sptk in emoji_dict:
		# 遍历所有可能的标签（emoji_dict包含所有标签）
		sptk_dict[sptk] = s.count(sptk)
		# 统计当前标签在文本中出现的次数，存储到字典中
		s = s.replace(sptk, "")
		# 从文本中移除当前标签，保留纯文本内容
		# 注意：这里会移除所有标签，包括语言标签、情感标签、事件标签等
	
	emo = "<|NEUTRAL|>"
	# 初始化默认情感为中性（NEUTRAL）
	
	for e in emo_dict:
		# 遍历所有情感标签（emo_dict只包含情感相关的标签）
		if sptk_dict[e] > sptk_dict[emo]:
			# 如果当前情感标签的出现次数大于当前选择的情感
			emo = e
			# 更新为主要情感（选择出现次数最多的情感）
	
	for e in event_dict:
		# 遍历所有事件标签（event_dict只包含事件相关的标签）
		if sptk_dict[e] > 0:
			# 如果该事件标签在文本中出现过（次数>0）
			s = event_dict[e] + s
			# 将对应的事件表情符号添加到文本开头
			# 注意：如果有多个事件，只会添加第一个找到的事件表情
	
	s = s + emo_dict[emo]
	# 将主要情感的表情符号添加到文本末尾
	# 这样格式为：[事件表情]文本内容[情感表情]

	for emoji in emo_set.union(event_set):
		# 遍历所有情感和事件表情符号的并集
		# union()方法返回两个集合的并集（所有不重复的元素）
		s = s.replace(" " + emoji, emoji)
		# 移除表情符号前的空格，使表情符号紧贴文本
		s = s.replace(emoji + " ", emoji)
		# 移除表情符号后的空格，使表情符号紧贴文本
		# 这样可以让输出更紧凑，避免表情符号和文本之间有空格
	
	return s.strip()
	# 去除文本首尾的空白字符（空格、换行等）并返回

def format_str_v3(s):
    # 第三版格式化函数：处理多语言分段，合并相同情感和事件
    # 这是最完善的格式化函数，能够处理包含多语言切换的识别结果
    # 参数: s - 原始识别文本（可能包含多语言分段和多个标签）
    # 返回: 格式化后的统一文本（合并了相同的情感/事件，去除了重复标记）
    
	def get_emo(s):
		# 内部辅助函数：获取文本末尾的情感表情符号
		# 参数: s - 文本字符串
		# 返回: 如果文本最后一个字符是情感表情符号，返回该表情；否则返回None
		return s[-1] if s[-1] in emo_set else None
		# s[-1]表示字符串的最后一个字符
		# 检查该字符是否在情感表情符号集合中
	
	def get_event(s):
		# 内部辅助函数：获取文本开头的事件表情符号
		# 参数: s - 文本字符串
		# 返回: 如果文本第一个字符是事件表情符号，返回该表情；否则返回None
		return s[0] if s[0] in event_set else None
		# s[0]表示字符串的第一个字符
		# 检查该字符是否在事件表情符号集合中

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	# 首先处理特殊情况：将"无语音+未知事件"的组合标签替换为问号表情
	# 这需要在其他处理之前完成，避免被后续步骤拆分
	
	for lang in lang_dict:
		# 遍历语言标签字典
		s = s.replace(lang, "<|lang|>")
		# 将所有不同的语言标签统一替换为"<|lang|>"标记
		# 这样可以将多语言文本用统一的标记分隔，便于后续分段处理
	
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	# 按统一的语言标记"<|lang|>"分割文本，得到不同语言段的列表
	# 对每个语言段调用format_str_v2进行格式化（移除标签，添加表情符号）
	# strip(" ")去除每个段的首尾空格
	# 结果：s_list是一个列表，每个元素是一个格式化后的语言段
	
	new_s = " " + s_list[0]
	# 初始化新文本，从第一个语言段开始
	# 前面加一个空格，用于后续处理
	
	cur_ent_event = get_event(new_s)
	# 获取当前段的事件表情符号（如果存在）
	# 用于后续判断是否需要合并相同的事件标记
	
	for i in range(1, len(s_list)):
		# 遍历后续的语言段（从第二个开始）
		if len(s_list[i]) == 0:
			# 如果当前段为空，跳过处理
			continue
		
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			# 如果当前段的事件表情与上一段相同，且不为None
			# 说明连续两段有相同的事件标记（如连续的背景音乐）
			s_list[i] = s_list[i][1:]
			# 移除当前段开头的事件表情符号，避免重复显示
			# [1:]表示从第二个字符开始到末尾（跳过第一个字符）
		
		#else:
		# 注释掉的else分支，可能是之前的逻辑
		
		cur_ent_event = get_event(s_list[i])
		# 更新当前事件表情符号，用于下一次循环的比较
		
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			# 如果当前段的情感表情与已拼接文本的情感表情相同，且不为None
			# 说明连续两段有相同的情感标记（如连续的开心情绪）
			new_s = new_s[:-1]
			# 移除已拼接文本末尾的情感表情符号，避免重复显示
			# [:-1]表示从开头到倒数第二个字符（移除最后一个字符）
		
		new_s += s_list[i].strip().lstrip()
		# 将当前段拼接到新文本中
		# strip()去除首尾空白字符，lstrip()去除左侧空白字符（双重保险）
	
	new_s = new_s.replace("The.", " ")
	# 替换特定的错误模式"The."为空格
	# 这可能是ASR模型在某些情况下的识别错误，需要后处理修正
	
	return new_s.strip()
	# 去除最终文本首尾的空白字符并返回
	# 这是格式化后的最终结果


# ==================== 模型初始化 ====================
model_asr = AutoModel(
    # 自动语音识别（ASR）模型初始化
    # 使用FunASR的AutoModel类加载SenseVoice模型
    model="/home/octopus/data/llm_list/SenseVoiceSmall",
    # 模型名称：SenseVoice小模型（iic是模型发布者的命名空间）
    # Small版本是轻量级版本，适合实时识别场景
    trust_remote_code=True,
    # 信任远程代码：允许从模型仓库下载并执行自定义Python代码
    # 某些模型需要自定义的前处理或后处理代码
    remote_code="./model.py",
    # 远程代码文件路径：指定本地模型代码文件
    # 如果模型仓库中有自定义代码，会使用此文件
    device="cuda:0",
    # 设备：使用第一个GPU（CUDA设备编号0）
    # 如果系统有多个GPU，可以指定其他编号（如"cuda:1"）
    # 如果没有GPU，应改为"cpu"
    disable_update=True
    # 禁用模型更新：防止自动下载新版本模型
    # 确保使用固定版本的模型，避免版本不一致导致的问题
)

model_vad = AutoModel(
    # 语音活动检测（VAD）模型初始化
    # VAD用于检测音频中哪些部分是语音，哪些部分是静音或噪声
    model="/home/octopus/data/llm_list/fsmn_vad",
    # 模型名称：FSMN-VAD（Feedforward Sequential Memory Network VAD）
    # FSMN是一种高效的神经网络结构，适合实时VAD任务
    model_revision="v2.0.4",
    # 模型版本：指定使用v2.0.4版本
    # 确保使用特定版本的模型，避免版本差异
    disable_pbar = True,
    # 禁用进度条：模型加载时不显示进度条
    # 减少控制台输出，保持日志清洁
    max_end_silence_time=500,
    # 最大结束静音时间（单位：毫秒）
    # 当检测到静音超过500ms时，判定为语音段结束
    # 这个参数影响VAD对语音结束的判断敏感度
    # speech_noise_thres=0.6,
    # 语音噪声阈值（已注释）：用于区分语音和噪声
    # 默认值0.6，值越大越严格（更容易将语音误判为噪声）
    disable_update=True
    # 禁用模型更新：防止自动下载新版本模型
)

# ==================== 说话人注册文件列表 ====================
reg_spks_files = [
    # 已注册说话人的音频文件路径列表
    # 这些文件用作说话人验证的参考样本
    # 当启用说话人验证时，会将输入音频与这些参考样本进行比对
    "speaker/speaker1_a_cn_16k.wav"
    # 说话人1的音频文件
    # 文件名格式：speaker{编号}_{标识}_{语言}_{采样率}.wav
    # 这里表示：说话人1，标识a，中文，16kHz采样率
    # 可以添加更多说话人文件到此列表
]

def reg_spk_init(files):
    # 初始化已注册说话人数据
    # 参数: files - 说话人音频文件路径列表
    # 返回: 说话人数据字典 {说话人名称: {data: 音频数据数组, sr: 采样率}}
    
    reg_spk = {}
    # 初始化空字典，用于存储说话人数据
    
    for f in files:
        # 遍历每个说话人音频文件
        try:
            # 使用try-except捕获单个文件加载失败的情况
            # 这样即使某个文件加载失败，也不会影响其他文件的加载
            if not os.path.exists(f):
                # 检查文件是否存在
                logger.warning(f"Speaker file not found: {f}, skipping...")
                # 记录警告日志，提示文件不存在
                continue
                # 跳过不存在的文件，继续处理下一个
            
            data, sr = sf.read(f, dtype="float32")
            # 使用soundfile库读取音频文件
            # data: 音频数据，numpy数组，dtype="float32"表示32位浮点数格式
            # sr: 采样率（samples per second），如16000表示16kHz
            # float32格式的音频数据范围通常在[-1.0, 1.0]之间
            
            k, _ = os.path.splitext(os.path.basename(f))
            # 提取文件名（不含扩展名）作为说话人标识
            # os.path.basename(f): 获取文件名（不含路径），如"speaker1_a_cn_16k.wav"
            # os.path.splitext(): 分割文件名和扩展名，返回(文件名, 扩展名)
            # k: 文件名部分（不含扩展名），如"speaker1_a_cn_16k"
            # _: 扩展名部分（这里不需要，用下划线忽略）
            
            reg_spk[k] = {
                "data": data,  # 音频数据数组
                "sr":   sr,    # 采样率
            }
            # 将说话人数据存储到字典中，键为说话人名称，值为包含音频数据和采样率的字典
            
            logger.info(f"Successfully loaded speaker file: {f}")
            # 记录成功加载的日志
        
        except Exception as e:
            # 捕获加载过程中的任何异常（文件损坏、格式不支持等）
            logger.error(f"Failed to load speaker file {f}: {e}")
            # 记录错误日志，包含文件路径和异常信息
            continue
            # 继续处理下一个文件，不中断整个初始化过程
    
    return reg_spk
    # 返回说话人数据字典

try:
    # 尝试初始化说话人数据
    reg_spks = reg_spk_init(reg_spks_files)
    # 调用初始化函数，加载所有说话人文件
    
    if not reg_spks:
        # 如果字典为空（没有成功加载任何说话人文件）
        logger.warning("No valid speaker files loaded. Speaker verification will not work.")
        # 记录警告日志，提示说话人验证功能将不可用
except Exception as e:
    # 捕获初始化过程中的异常（如函数调用失败）
    logger.error(f"Failed to initialize speaker files: {e}")
    # 记录错误日志
    reg_spks = {}
    # 初始化为空字典，确保程序可以继续运行（说话人验证功能不可用）


def asr(audio, lang, cache, use_itn=False):
    # 自动语音识别函数
    # 参数: audio - 音频数据（numpy数组，float32格式，范围[-1.0, 1.0]）
    #      lang - 语言代码字符串（如"zh"中文、"en"英文、"auto"自动检测）
    #      cache - ASR缓存字典（用于流式识别，保持上下文状态）
    #      use_itn - 是否使用逆文本规范化（Inverse Text Normalization），默认False
    #                 ITN将数字、日期等从文本形式转换为标准形式（如"123"转为"一百二十三"）
    # 返回: 识别结果列表，每个元素包含识别文本和元数据
    
    # 注释掉的调试代码：将音频数据写入文件用于调试
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    
    start_time = time.time()
    # 记录ASR处理开始时间，用于性能统计
    
    result = model_asr.generate(
        # 调用ASR模型进行语音识别
        input           = audio,
        # 输入音频数据（numpy数组）
        cache           = cache,
        # 缓存字典：用于流式识别，保持模型内部状态
        # 在连续识别中，缓存可以帮助模型理解上下文，提高识别准确性
        language        = lang.strip(),
        # 语言代码：去除首尾空格后的语言标识
        # 支持的语言：zh（中文）、en（英文）、yue（粤语）、ja（日语）、ko（韩语）、auto（自动）
        use_itn         = use_itn,
        # 是否使用逆文本规范化：将识别结果中的数字、日期等转换为标准格式
        batch_size_s    = 60,
        # 批处理大小（单位：秒）：模型内部批处理的音频时长
        # 60秒表示每次处理最多60秒的音频，用于优化GPU利用率
    )
    
    end_time = time.time()
    # 记录ASR处理结束时间
    
    elapsed_time = end_time - start_time
    # 计算处理耗时（秒）
    
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    # 记录ASR处理耗时日志（转换为毫秒，保留2位小数）
    # DEBUG级别日志，用于性能分析和调试
    
    return result
    # 返回识别结果：通常是列表格式，包含识别文本、时间戳、置信度等信息

# ==================== FastAPI应用初始化 ====================
app = FastAPI()
# 创建FastAPI应用实例
# FastAPI是一个现代、快速的Web框架，用于构建API服务
# 支持异步处理、自动API文档生成等功能

app.add_middleware(
    # 添加中间件：在请求处理前后执行额外逻辑
    CORSMiddleware,
    # CORS（Cross-Origin Resource Sharing）中间件
    # 用于处理浏览器的跨域请求限制
    allow_origins=["*"],
    # 允许的来源：["*"]表示允许所有域名访问
    # 生产环境应限制为特定域名，如["https://example.com"]
    allow_credentials=True,
    # 允许携带凭证：允许请求携带cookies、认证信息等
    # 当allow_origins为["*"]时，此选项通常应设为False（浏览器限制）
    allow_methods=["*"],
    # 允许的HTTP方法：["*"]表示允许所有方法（GET、POST、PUT、DELETE等）
    allow_headers=["*"],
    # 允许的请求头：["*"]表示允许所有请求头
)

@app.exception_handler(Exception)
# 全局异常处理器装饰器：捕获所有未处理的异常
# 当应用中出现任何未捕获的异常时，会调用此函数
async def custom_exception_handler(request: Request, exc: Exception):
    # 自定义异常处理函数（异步函数）
    # 参数: request - HTTP请求对象，包含请求的详细信息
    #      exc - 异常对象，包含异常的类型和消息
    
    logger.error("Exception occurred", exc_info=True)
    # 记录错误日志，exc_info=True会包含完整的异常堆栈信息
    # 这对于调试和问题追踪非常重要
    
    if isinstance(exc, HTTPException):
        # 如果是FastAPI的HTTP异常（主动抛出的HTTP错误）
        status_code = exc.status_code
        # 获取异常中定义的HTTP状态码（如404、400等）
        message = exc.detail
        # 获取异常详情消息
        data = ""
        # 数据字段为空（错误响应通常不需要数据）
    elif isinstance(exc, RequestValidationError):
        # 如果是请求验证错误（Pydantic模型验证失败）
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        # 使用422状态码：请求格式正确但语义错误（验证失败）
        message = "Validation error: " + str(exc.errors())
        # 构造验证错误消息，包含所有验证失败的详细信息
        data = ""
        # 数据字段为空
    else:
        # 其他未知异常（未预期的错误）
        status_code = 500
        # 使用500状态码：内部服务器错误
        message = "Internal server error: " + str(exc)
        # 构造错误消息，包含异常字符串表示
        data = ""
        # 数据字段为空

    return JSONResponse(
        # 返回JSON格式的错误响应
        status_code=status_code,
        # HTTP状态码
        content=TranscriptionResponse(
            # 使用标准响应格式
            code=status_code,
            # 响应代码（与HTTP状态码相同）
            info=message,
            # 错误信息
            data=data
            # 数据字段（错误时为空）
        ).model_dump()
        # 将Pydantic模型转换为字典（JSONResponse需要字典格式）
    )

# ==================== 响应模型定义 ====================
class TranscriptionResponse(BaseModel):
    # 转录响应数据模型
    # 所有WebSocket响应和HTTP错误响应都使用此格式
    # 继承自Pydantic的BaseModel，提供自动验证和序列化
    
    code: int
    # 状态码（整数类型）
    # 0: 成功（识别结果）
    # 2: 检测到语音/说话人（中间状态）
    # 400: 客户端错误（如配置错误）
    # 500: 服务器错误（如处理异常）
    
    info: str
    # 信息字段（字符串类型）
    # 通常包含JSON格式的详细识别结果或错误信息
    # 对于成功响应，可能包含完整的ASR结果元数据
    
    data: str
    # 数据字段（字符串类型）
    # 对于成功响应：包含格式化后的识别文本
    # 对于错误响应：通常为空字符串
    # 对于说话人检测：可能包含说话人名称

# ==================== WebSocket端点 ====================
@app.websocket("/ws/transcribe")
# WebSocket路由装饰器：定义WebSocket端点路径
# 客户端通过 ws://host/ws/transcribe 连接到此端点
async def websocket_endpoint(websocket: WebSocket):
    # WebSocket连接处理函数（异步函数）
    # 参数: websocket - WebSocket连接对象，用于接收和发送消息
    
    audio_buffer = np.array([], dtype=np.float32)
    # 音频缓冲区：存储接收到的原始音频数据
    # dtype=np.float32：32位浮点数格式，范围通常在[-1.0, 1.0]
    # 初始化为空数组
    
    audio_vad = np.array([], dtype=np.float32)
    # VAD音频缓冲区：存储用于VAD（语音活动检测）处理的音频数据
    # 这个缓冲区会累积音频，直到检测到完整的语音段
    
    cache = {}
    # VAD模型缓存：用于流式VAD处理，保持模型内部状态
    # 在连续处理中，缓存帮助VAD模型理解上下文，提高检测准确性
    
    cache_asr = {}
    # ASR模型缓存：用于流式ASR处理，保持模型内部状态
    # 在连续识别中，缓存帮助ASR模型理解上下文，提高识别准确性
    
    try:
        # 主处理逻辑（使用try-except捕获异常）
        query_params = parse_qs(websocket.scope['query_string'].decode())
        # 解析WebSocket连接URL的查询参数
        # websocket.scope['query_string']: 获取URL查询字符串（字节格式）
        # .decode(): 将字节转换为字符串
        # parse_qs(): 解析查询字符串，返回字典格式
        # 例如：ws://host/ws/transcribe?sv=true&lang=zh
        #      解析后：{'sv': ['true'], 'lang': ['zh']}
        
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        # 获取说话人验证参数（sv）
        # query_params.get('sv', ['false']): 获取'sv'参数，默认值为['false']
        # [0]: 取列表第一个元素（parse_qs返回的是列表）
        # .lower(): 转换为小写
        # in ['true', '1', 't', 'y', 'yes']: 检查是否为真值
        # 如果参数值为这些值之一，则启用说话人验证
        
        lang = query_params.get('lang', ['auto'])[0].lower()
        # 获取语言参数（lang），默认值为'auto'（自动检测）
        # 支持的语言：zh（中文）、en（英文）、yue（粤语）、ja（日语）、ko（韩语）、auto（自动）
        
        # 验证说话人验证配置
        if sv and not reg_spks:
            # 如果启用了说话人验证但没有加载说话人文件
            await websocket.accept()
            # 先接受连接（必须在发送消息前接受连接）
            error_response = TranscriptionResponse(
                code=400,
                # 400状态码：客户端错误（配置错误）
                info="Speaker verification is enabled but no speaker files are loaded",
                # 错误信息：说明说话人验证已启用但未加载说话人文件
                data=""
                # 数据字段为空
            )
            await websocket.send_json(error_response.model_dump())
            # 发送错误消息（JSON格式）
            await websocket.close()
            # 关闭连接
            return
            # 退出函数，不再处理后续逻辑
        
        await websocket.accept()
        # 接受WebSocket连接（正常情况）
        # 必须在接收或发送消息前调用此方法
        
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        # 计算音频块大小（样本数）
        # config.chunk_size_ms: 块大小（毫秒），如300ms
        # config.sample_rate: 采样率（Hz），如16000Hz
        # 计算：300ms * 16000Hz / 1000 = 4800个样本
        # int(): 转换为整数（样本数必须是整数）
        
        if chunk_size <= 0:
            # 如果块大小无效（配置错误导致）
            error_response = TranscriptionResponse(
                code=500,
                # 500状态码：服务器错误（配置错误）
                info="Invalid chunk size configuration",
                # 错误信息：块大小配置无效
                data=""
            )
            await websocket.send_json(error_response.model_dump())
            # 发送错误消息
            await websocket.close()
            # 关闭连接
            return
            # 退出函数

        last_vad_beg = last_vad_end = -1
        # VAD检测到的语音段边界（单位：毫秒）
        # last_vad_beg: 语音开始时间戳（相对于VAD缓冲区的开始）
        # last_vad_end: 语音结束时间戳
        # -1表示尚未检测到（初始值）
        
        offset = 0
        # 时间偏移量（单位：毫秒）
        # 用于处理流式音频的时间戳累积
        # 每次处理完一个语音段后，offset会增加，用于调整后续段的时间戳
        
        hit = False
        # 说话人验证匹配标志
        # True: 已匹配到已注册的说话人
        # False: 尚未匹配或匹配失败
        # 注意：在当前代码版本中，说话人验证相关逻辑已被简化
        
        speech_detected_sent = False
        # 是否已发送"检测到语音"消息的标志
        # 防止重复发送检测消息（每个语音段只发送一次）
        
        buffer = b""
        # 原始字节缓冲区：存储接收到的二进制数据
        # b""表示空字节字符串
        # 用于处理可能不完整的音频数据包（确保按样本对齐）
        while True:
            # 主循环：持续接收和处理音频数据
            # 直到WebSocket连接断开或发生错误
            try:
                data = await websocket.receive_bytes()
                # 异步接收WebSocket二进制数据（音频流）
                # await: 等待数据到达（非阻塞）
                # receive_bytes(): 接收二进制数据，返回bytes对象
            except Exception as e:
                # 接收数据时发生异常（连接断开、网络错误等）
                logger.error(f"Error receiving data: {e}")
                # 记录错误日志
                error_response = TranscriptionResponse(
                    code=500,
                    # 500状态码：服务器错误
                    info=f"Error receiving audio data: {str(e)}",
                    # 错误信息：包含异常详情
                    data=""
                )
                try:
                    await websocket.send_json(error_response.model_dump())
                    # 尝试发送错误消息给客户端
                except:
                    pass
                    # 如果发送失败（连接已断开），忽略异常
                break
                # 退出循环，结束处理
            
            buffer += data
            # 将接收到的数据追加到字节缓冲区
            # += 操作符用于拼接字节数据
            
            if len(buffer) < 2:
                # 如果缓冲区数据不足2字节（一个int16样本需要2字节）
                continue
                # 继续接收更多数据，不进行后续处理
                
            try:
                audio_buffer = np.append(
                    audio_buffer,
                    # 将新数据追加到音频缓冲区
                    np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
                    # 从字节缓冲区转换为numpy数组：
                    # buffer[:len(buffer) - (len(buffer) % 2)]: 确保字节数是2的倍数
                    #   len(buffer) % 2: 计算余数（不足一个样本的字节数）
                    #   减去余数，只处理完整的样本（int16需要2字节）
                    # dtype=np.int16: 解释为16位有符号整数（-32768到32767）
                    # .astype(np.float32): 转换为32位浮点数
                    # / 32767.0: 归一化到[-1.0, 1.0]范围
                    #   32767是16位有符号整数的最大值
                )
            except Exception as e:
                # 音频处理异常（数据格式错误、内存不足等）
                logger.error(f"Error processing audio buffer: {e}")
                # 记录错误日志
                error_response = TranscriptionResponse(
                    code=500,
                    # 500状态码：服务器错误
                    info=f"Error processing audio data: {str(e)}",
                    # 错误信息
                    data=""
                )
                try:
                    await websocket.send_json(error_response.model_dump())
                    # 尝试发送错误消息
                except:
                    pass
                    # 如果发送失败，忽略异常
                continue
                # 继续处理下一批数据（不退出循环）
                
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
            # 保留缓冲区中未处理的字节（不足一个样本的部分）
            # 这些字节会在下次接收数据时与新区块一起处理
            # 例如：如果buffer有5字节，处理4字节，保留1字节
   
            while len(audio_buffer) >= chunk_size:
                # 当音频缓冲区有足够数据时，处理一个块
                # 这个内层循环确保只要有足够数据就持续处理
                chunk = audio_buffer[:chunk_size]
                # 提取一个块的数据（从开头取chunk_size个样本）
                audio_buffer = audio_buffer[chunk_size:]
                # 从缓冲区移除已处理的数据（保留chunk_size之后的数据）
                audio_vad = np.append(audio_vad, chunk)
                # 将块追加到VAD缓冲区
                # VAD缓冲区会累积音频，直到检测到完整的语音段
                    
                if last_vad_beg > 1 and not speech_detected_sent:
                    # 如果已检测到语音开始（last_vad_beg > 1），且尚未发送检测消息
                    # 注意：这里简化了说话人验证逻辑，直接发送语音检测消息
                    response = TranscriptionResponse(
                        code=2,
                        # 状态码2：检测到语音（中间状态）
                        info="detect speech",
                        # 信息：检测到语音
                        data=''
                        # 数据字段为空（此时还没有识别结果）
                    )
                    await websocket.send_json(response.model_dump())
                    # 发送语音检测消息给客户端
                    speech_detected_sent = True
                    # 标记已发送检测消息，防止重复发送

                try:
                    res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                    # 调用VAD模型检测语音活动
                    # input=chunk: 输入音频块
                    # cache=cache: VAD缓存（保持模型状态）
                    # is_final=False: 表示这是流式处理，不是最终块
                    # chunk_size=config.chunk_size_ms: 块大小（毫秒），用于VAD内部处理
                except Exception as e:
                    # VAD处理异常（模型错误、内存不足等）
                    logger.error(f"Error in VAD processing: {e}")
                    # 记录错误日志
                    error_response = TranscriptionResponse(
                        code=500,
                        # 500状态码：服务器错误
                        info=f"VAD processing error: {str(e)}",
                        # 错误信息
                        data=""
                    )
                    try:
                        await websocket.send_json(error_response.model_dump())
                        # 尝试发送错误消息
                    except:
                        pass
                        # 如果发送失败，忽略异常
                    continue
                    # 继续处理下一块（不退出循环）
                    
                # logger.info(f"vad inference: {res}")
                # 注释掉的VAD推理日志（用于调试）
                
                if len(res) > 0 and len(res[0]["value"]):
                    # 如果VAD返回有效结果
                    # res通常是列表格式：[{"value": [[开始时间, 结束时间], ...], ...}]
                    vad_segments = res[0]["value"]
                    # 提取语音段列表：每个元素是[开始时间(ms), 结束时间(ms)]
                    # 例如：[[100, 500], [800, 1200]] 表示两个语音段
                    
                    for segment in vad_segments:
                        # 遍历每个语音段
                        if segment[0] > -1:
                            # 如果检测到语音开始（-1表示未检测到）
                            last_vad_beg = segment[0]
                            # 更新语音开始时间戳（毫秒）
                            # 这个时间戳是相对于当前VAD缓冲区的开始位置
                            
                        if segment[1] > -1:
                            # 如果检测到语音结束
                            last_vad_end = segment[1]
                            # 更新语音结束时间戳（毫秒）
                            
                        if last_vad_beg > -1 and last_vad_end > -1:
                            # 如果同时检测到开始和结束，说明有一个完整的语音段
                            last_vad_beg -= offset
                            # 调整开始时间为相对时间（减去已处理的偏移量）
                            # 因为VAD返回的时间戳是相对于VAD缓冲区的，需要转换为绝对时间
                            
                            last_vad_end -= offset
                            # 调整结束时间为相对时间
                            
                            offset += last_vad_end
                            # 更新偏移量：累加已处理的音频时长
                            # 下次处理时，新段的时间戳需要减去这个偏移量
                            
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            # 将开始时间（毫秒）转换为样本索引
                            # 例如：100ms * 16000Hz / 1000 = 1600个样本
                            
                            end = int(last_vad_end * config.sample_rate / 1000)
                            # 将结束时间（毫秒）转换为样本索引
                            
                            # 边界检查
                            if beg < 0 or end < beg or end > len(audio_vad):
                                # 如果索引无效（负数、结束小于开始、超出缓冲区长度）
                                logger.warning(f"Invalid VAD segment: beg={beg}, end={end}, audio_vad_len={len(audio_vad)}")
                                # 记录警告日志
                                audio_vad = audio_vad[max(0, end):] if end < len(audio_vad) else np.array([], dtype=np.float32)
                                # 清理无效数据：
                                # 如果end在有效范围内，保留end之后的数据
                                # 否则清空VAD缓冲区
                                last_vad_beg = last_vad_end = -1
                                # 重置VAD边界
                                hit = False
                                # 重置说话人验证标志
                                speech_detected_sent = False
                                # 重置检测消息标志
                                continue
                                # 跳过当前段，继续处理下一个
                            
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            # 记录语音段长度日志（样本数）
                            
                            try:
                                result = None if sv and not hit else asr(audio_vad[beg:end], lang.strip(), cache_asr, True)
                                # 进行ASR识别：
                                # 如果启用了说话人验证且未匹配到说话人，跳过ASR（result=None）
                                # 否则，对语音段进行ASR识别
                                # audio_vad[beg:end]: 提取语音段的音频数据（numpy数组切片）
                                # lang.strip(): 语言代码（去除空格）
                                # cache_asr: ASR缓存（保持上下文）
                                # True: 使用逆文本规范化（ITN）
                                logger.info(f"asr response: {result}")
                                # 记录ASR结果日志
                            except Exception as e:
                                # ASR处理异常（模型错误、GPU内存不足等）
                                logger.error(f"Error in ASR processing: {e}")
                                # 记录错误日志
                                error_response = TranscriptionResponse(
                                    code=500,
                                    # 500状态码：服务器错误
                                    info=f"ASR processing error: {str(e)}",
                                    # 错误信息
                                    data=""
                                )
                                try:
                                    await websocket.send_json(error_response.model_dump())
                                    # 尝试发送错误消息
                                except:
                                    pass
                                    # 如果发送失败，忽略异常
                                result = None
                                # 设置结果为None，后续不会发送识别结果
                            
                            audio_vad = audio_vad[end:]
                            # 从VAD缓冲区移除已处理的音频数据
                            # 只保留end之后的数据，用于后续处理
                            
                            last_vad_beg = last_vad_end = -1
                            # 重置VAD边界（准备检测下一个语音段）
                            
                            hit = False
                            # 重置说话人验证标志（每次ASR后重置）
                            
                            speech_detected_sent = False
                            # 重置检测消息标志（准备发送下一个语音段的检测消息）
                            
                            if result is not None:
                                # 如果ASR识别成功（result不为None）
                                try:
                                    formatted_text = format_str_v3(result[0]['text']) if result and len(result) > 0 and 'text' in result[0] else ""
                                    # 格式化识别文本：
                                    # 1. 检查result是否有效（不为None且长度>0）
                                    # 2. 检查result[0]中是否有'text'字段
                                    # 3. 如果有，提取result[0]['text']并使用format_str_v3格式化
                                    # 4. 如果无效，返回空字符串
                                    # format_str_v3会处理多语言、情感、事件等标签
                                    
                                    response = TranscriptionResponse(
                                        code=0,
                                        # 状态码0：成功（识别结果）
                                        info=json.dumps(result[0], ensure_ascii=False) if result and len(result) > 0 else "",
                                        # info字段：完整的ASR结果（JSON格式）
                                        # json.dumps(): 将字典转换为JSON字符串
                                        # ensure_ascii=False: 允许非ASCII字符（如中文）直接输出
                                        # result[0]通常包含：text（文本）、timestamp（时间戳）、confidence（置信度）等
                                        data=formatted_text
                                        # data字段：格式化后的识别文本（用户友好的格式）
                                    )
                                    await websocket.send_json(response.model_dump())
                                    # 发送识别结果给客户端（JSON格式）
                                except Exception as e:
                                    # 格式化或发送结果时异常（JSON序列化错误、网络错误等）
                                    logger.error(f"Error formatting or sending ASR result: {e}")
                                    # 记录错误日志
                                    error_response = TranscriptionResponse(
                                        code=500,
                                        # 500状态码：服务器错误
                                        info=f"Error formatting result: {str(e)}",
                                        # 错误信息
                                        data=""
                                    )
                                    try:
                                        await websocket.send_json(error_response.model_dump())
                                        # 尝试发送错误消息
                                    except:
                                        pass
                                        # 如果发送失败，忽略异常
                                
                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')
                        # 注释掉的调试日志（用于调试VAD边界和缓冲区状态）

    except WebSocketDisconnect:
        # 捕获WebSocket正常断开异常
        # 这是FastAPI在客户端主动断开连接时抛出的异常
        logger.info("WebSocket disconnected")
        # 记录断开日志（INFO级别，正常情况）
    except Exception as e:
        # 捕获其他未预期的异常（主循环中的异常）
        logger.error(f"Unexpected error in WebSocket endpoint: {e}\nCall stack:\n{traceback.format_exc()}")
        # 记录错误日志，包含异常信息和完整的堆栈跟踪
        # traceback.format_exc(): 获取格式化的异常堆栈信息
        try:
            error_response = TranscriptionResponse(
                code=500,
                # 500状态码：内部服务器错误
                info=f"Internal server error: {str(e)}",
                # 错误信息：包含异常详情
                data=""
            )
            await websocket.send_json(error_response.model_dump())
            # 尝试发送错误消息给客户端
        except:
            pass
            # 如果发送失败（连接已断开），忽略异常
        try:
            await websocket.close()
            # 尝试关闭WebSocket连接
        except:
            pass
            # 如果关闭失败（连接已断开），忽略异常
    finally:
        # finally块：无论是否发生异常都会执行
        # 用于清理资源，确保内存和状态被正确释放
        audio_buffer = np.array([], dtype=np.float32)
        # 清空音频缓冲区，释放内存
        
        audio_vad = np.array([], dtype=np.float32)
        # 清空VAD缓冲区，释放内存
        
        cache.clear()
        # 清空VAD模型缓存，释放内存
        
        cache_asr.clear()
        # 清空ASR模型缓存，释放内存
        
        logger.info("Cleaned up resources after WebSocket disconnect")
        # 记录清理完成日志

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=8034, help='Port number to run the FastAPI app on.')
    
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
