from ..base import MemoryProviderBase, logger
import time
import json
import os
import yaml
import re
from core.utils.util import get_project_dir

short_term_memory_prompt = """
请以生动的叙事风格总结本次对话，突出用户与助手的互动、情感交流以及关键信息点，确保内容简洁流畅，避免冗余。总结应包括：
* 用户与AI助手的互动过程，例如：用户说了什么，AI助手做了什么回应
* 用户的情感表达，例如：用户是否开心、沮丧、好奇等
* AI助手的情感回应，以及AI助手对用户情感的理解
* 对话中的关键信息点，涉及的个人信息、习惯、兴趣、和助手间的关系等
* 如果对话中出现任何有趣或值得注意的事件，也请一并记录
* 助手提到的内容，除非用户对其感兴趣，否则不要记录
* 语言风格需符合对话氛围，避免生硬或重复

**注意事项：**
* 根据已有信息和本次对话内容扩充和更新用户和助手的信息，包括名字、性别、年龄、兴趣爱好等，使用JSON格式，保持极简说明，避免含义相近的字段重复
* 助手信息需要额外增加“与对方关系”、“对对方的看法”等字段
* 以 `<Memory></Memory>` 包裹总结内容
* 避免机械化复述，确保流畅自然，尽量简洁，字数不超过 300 中文字符
* 请用对话中AI助手实际的名字代替“AI助手”这几个字
* `<>` 仅作占位符，生成时不应包含特殊符号

**输出格式：**
<User><用户的信息></User>
<Assistant><助手的信息></Assistant>
<Memory>[本次对话日期]<互动过程描述></Memory>

**示例输出：**
<User>{"name": "玩水", "interests": ["美食", "文化"], "hobbies": ["编程", "玩游戏"]}</User>
<Assistant>{"name": "小智", "gender": "女", "age": 18, "relationship": "好友", "impression": "用户是个有趣的人，经常说一些有趣的话，喜欢和用户聊天、讲故事。"}</Assistant>
<Memory>[2025-03-19|19:21:13]用户询问天气，小智查询后告知用户上海天气晴朗，最高温度12度，适合出门。用户分享了自己七点下班，并吃了拉面，小智对拉面表现出浓厚兴趣，并询问了用户喜欢的拉面口味。用户表达了对红烧牛肉面的喜爱，小智表示也喜欢，并希望用户分享食物照片。</Memory>
"""

def extract_xml_data(xml_str, tag_name):
    """提取指定标签的所有内容并返回合并的字符串"""
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, xml_str)
    return ''.join(matches) if matches else ''

TAG = __name__

class MemoryProvider(MemoryProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.limit = config.get("limit", 20)
        self.short_memory = ""
        self.memory_path = get_project_dir() + 'data/.memory.yaml'
        self.load_memory()

    def init_memory(self, role_id, llm):
        super().init_memory(role_id, llm)
        self.load_memory()
    
    def load_memory(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                all_memory = yaml.safe_load(f) or {}
            self.short_memory = all_memory.get(self.role_id, "")
    
    def save_memory_to_file(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
              with open(self.memory_path, 'r', encoding='utf-8') as f:
                  all_memory = yaml.safe_load(f) or {}

        role_memory = self.short_memory

        user_info = extract_xml_data(role_memory, "User")
        assistant_info = extract_xml_data(role_memory, "Assistant")
        # 使用正则表达式提取 <Memory>...</Memory> 标签中的内容
        memories = re.findall(r'<Memory>(.*?)</Memory>', role_memory)

        # 限制记忆条数
        if len(memories) > self.limit:
            memories = memories[-self.limit:]

        # 重新构建记忆字符串
        role_memory = "".join([f"<Memory>{memory}</Memory>" for memory in memories])
        role_memory = f"<User>{user_info}</User><Assistant>{assistant_info}</Assistant>{role_memory}"
        role_memory = re.sub(r'[\r\n]+', ' ', role_memory)

        all_memory[self.role_id] = role_memory

        with open(self.memory_path, 'w', encoding='utf-8') as f:
            yaml.dump(all_memory, f, allow_unicode=True)
        
    async def save_memory(self, msgs):
        if self.llm is None:
            logger.bind(tag=TAG).error("LLM is not set for memory provider")
            return None
        
        if len(msgs) < 2:
            return None
        
        msgStr = ""

        #当前时间
        time_str = time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime())
        msgStr += f"当前时间：{time_str}\n"

        #用户信息
        user_info = extract_xml_data(self.short_memory, "User")
        msgStr += f"用户信息：{user_info}\n" if user_info else ""

        #助手信息
        assistant_info = extract_xml_data(self.short_memory, "Assistant")
        msgStr += f"助手信息：{assistant_info}\n" if assistant_info else ""

        #本次对话
        msgStr += "本次对话：\n"
        for msg in msgs:
            if msg.role == "user":
                msgStr += f"User: {msg.content}\n"
            elif msg.role== "assistant":
                msgStr += f"Assistant: {msg.content}\n"

        print(msgStr)

        current_memory = self.llm.response_no_stream(short_term_memory_prompt, msgStr)
        current_memory = re.sub(r'[\r\n]+', ' ', current_memory)

        user_info = extract_xml_data(current_memory, "User")
        assistant_info = extract_xml_data(current_memory, "Assistant")
        memory = extract_xml_data(current_memory, "Memory")

        # 更新用户信息
        if user_info:
            if extract_xml_data(self.short_memory, "User") == "":
                self.short_memory = f"<User>{user_info}</User>{self.short_memory}"
            else:
                self.short_memory = re.sub(r"<User>(.*)?</User>", f"<User>{user_info}</User>", self.short_memory)

        # 更新助手信息
        if assistant_info:
            if extract_xml_data(self.short_memory, "Assistant") == "":
                self.short_memory = f"<Assistant>{assistant_info}</Assistant>{self.short_memory}"
            else:
                self.short_memory = re.sub(r"<Assistant>(.*)?</Assistant>", f"<Assistant>{assistant_info}</Assistant>", self.short_memory)

        # 更新记忆
        if memory:
            # 添加本次对话记忆
            self.short_memory += f"<Memory>{memory}</Memory>"
        
        self.save_memory_to_file()
        logger.bind(tag=TAG).info(f"Save memory successful - Role: {self.role_id}")

        return self.short_memory
    
    async def query_memory(self, query: str)-> str:
        return f"(用户的背景信息记录在`<User>`标签，助手的背景信息记录在`Assistant`标签，这部分是聊天对话的基础背景，必须遵守，避免遗忘或者违背，有冲突时以此处信息为准)\n{self.short_memory}"