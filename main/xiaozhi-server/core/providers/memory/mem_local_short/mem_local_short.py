from ..base import MemoryProviderBase, logger
import time
import json
import os
import yaml
import re
from core.utils.util import get_project_dir

short_term_memory_prompt = """
请以叙事风格总结本次对话，突出用户与AI助手的互动、情感交流、以及关键信息点，总结内容应包括：
* 用户与AI助手的互动过程，例如：用户说了什么，AI助手做了什么回应。
* 用户的情感表达，例如：用户是否开心、沮丧、好奇等。
* AI助手的情感回应，以及AI助手对用户情感的理解。
* 对话中的关键信息点，例如：用户的名字、喜欢的食物、音乐、以及其他个人偏好。
* AI助手在对话中展现的个性特点。
* 如果对话中出现任何有趣或值得注意的事件，也请一并记录。

注意事项：
* 输出的总结以<Memory></Memory>包裹。
* 避免重复性的内容和用词，总长度不超过800中文字符。
* 请用对话中AI助手实际的名字代替“AI助手”这几个字。
* 尖括号<>是内容指示占位符，输出结果里不要出现特殊符号。

输出格式：
<Memory>[本次对话日期]<互动过程描述></Memory>

示例输出：
<Memory>[2025-03-19 19:21:13]用户询问天气，小智查询后告知用户上海天气晴朗，最高温度12度，适合出门。用户分享了自己七点下班，并吃了拉面，小智对拉面表现出浓厚兴趣，并询问了用户喜欢的拉面口味。用户表达了对红烧牛肉面的喜爱，小智表示也喜欢，并希望用户分享照片。。</Memory>
"""

def extract_json_data(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start+1)
    #print("start:", start, "end:", end)
    if start == -1 or end == -1:
        try:
            jsonData = json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    jsonData = json_code[start+7:end]
    return jsonData

TAG = __name__

class MemoryProvider(MemoryProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.limit = config.get("limit", 20)
        self.short_momery = ""
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
        if self.role_id in all_memory:
            self.short_momery = all_memory[self.role_id]
    
    def save_memory_to_file(self):
        all_memory = {}
        if os.path.exists(self.memory_path):
              with open(self.memory_path, 'r', encoding='utf-8') as f:
                  all_memory = yaml.safe_load(f) or {}

        role_memory = json.loads(all_memory.get(self.role_id, ""))
        role_memory += self.short_momery

        print(role_memory)

        # 使用正则表达式提取 <Memory>...</Memory> 标签中的内容
        memories = re.findall(r'<Memory>(.*?)</Memory>', role_memory)

        # 限制记忆条数
        if len(memories) > self.limit:
            memories = memories[-self.limit:]

        # 重新构建记忆字符串
        role_memory = "".join([f"<Memory>{memory}</Memory>" for memory in memories])

        all_memory[self.role_id] = json.dumps(role_memory, ensure_ascii=False)

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

        for msg in msgs:
            if msg.role == "user":
                msgStr += f"User: {msg.content}\n"
            elif msg.role== "assistant":
                msgStr += f"Assistant: {msg.content}\n"

        self.short_momery = self.llm.response_no_stream(short_term_memory_prompt, msgStr)
        
        self.save_memory_to_file()
        logger.bind(tag=TAG).info(f"Save memory successful - Role: {self.role_id}")

        return self.short_momery
    
    async def query_memory(self, query: str)-> str:
        return self.short_momery