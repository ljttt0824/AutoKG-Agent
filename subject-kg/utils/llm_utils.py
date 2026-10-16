"""
Utility functions for interacting with LLMs (OpenAI API).
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional

import config
from openai import OpenAI, APIError

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with LLM APIs (OpenAI).
    """

    def __init__(self, model_name: str = config.DEFAULT_MODEL, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the model to use
            api_key: API key (if None, will use config or environment variable)
            api_base: API base URL (if None, will use config or default)
        """
        self.model_name = model_name
        self.api_key = api_key or config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or config.OPENAI_API_BASE

        if not self.api_key or self.api_key == "your-api-key-here":
            error_msg = "No valid API key provided. Please set your OpenAI API key in config.py or as an environment variable."
            logger.error(error_msg)
            # For testing purposes, use a mock response
            self.mock_mode = True
            logger.warning("Running in MOCK MODE. Will return mock responses instead of calling API.")
        else:
            self.mock_mode = False

        try:
            # Initialize OpenAI client
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=30.0,  # 30 seconds timeout
                max_retries=3  # Maximum number of retries
            )
            if not self.mock_mode:
                logger.info(f"Initialized LLM client with model {model_name}")
                logger.info(f"Using API base: {self.api_base}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.mock_mode = True
            logger.warning("Running in MOCK MODE due to initialization error.")

    def call(self, prompt: str, temperature: float = config.TEMPERATURE,
             max_tokens: int = config.MAX_TOKENS, top_p: float = config.TOP_P,
             response_format: str = "json") -> str:
        """
        Call the LLM API with a prompt.

        Args:
            prompt: Input prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p parameter for generation
            response_format: Format of the response ("json" or "text")

        Returns:
            Model response
        """
        # Check if in mock mode
        if self.mock_mode:
            logger.info("Using mock response instead of calling API")
            # Generate a simple mock response based on the prompt
            if "relationship" in prompt.lower():
                if response_format == "text":
                    mock_response = "The relationship indicates that IPv4 is a specific version or implementation of the Internet Protocol (IP)."
                else:
                    # Mock relationship response
                    mock_response = """[
                        {
                            "source": "Internet Protocol (IP)",
                            "target": "Internet protocol suite",
                            "type": "part_of"
                        },
                        {
                            "source": "IPv4",
                            "target": "Internet Protocol (IP)",
                            "type": "is_a"
                        },
                        {
                            "source": "IPv6",
                            "target": "Internet Protocol (IP)",
                            "type": "is_a"
                        }
                    ]"""
            else:
                if response_format == "text":
                    mock_response = "The Internet Protocol (IP) is the principal communications protocol in the Internet protocol suite for relaying packets across network boundaries."
                else:
                    # Mock entity response
                    mock_response = """[
                        {
                            "name": "Internet Protocol",
                            "type": "protocol",
                            "definition": "The principal communications protocol in the Internet protocol suite."
                        },
                        {
                            "name": "IPv4",
                            "type": "protocol",
                            "definition": "The fourth version of the Internet Protocol."
                        }
                    ]"""
            logger.info(f"Mock response: {mock_response}")
            return mock_response

        # Determine system message based on response format
        if response_format == "text":
            # System message for text response (used in entity_definer.py)
            system_message = """You are a computer network course instructor.
IMPORTANT: Your responses must be plain text.
- Provide clear and concise definitions
- Keep responses focused and direct
- No JSON formatting, just pure text
- No explanations or additional context beyond the definition"""
        else:
            # System message for JSON response (default)
            system_message = """You are a computer network course instructor.
IMPORTANT: Your responses must be valid JSON arrays.
- Start with '[' and end with ']'
- Use double quotes for strings
- Include all required fields
- No explanations or additional text
- Keep responses focused and concise"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"\nAPI Call Attempt {attempt + 1}:")
                logger.info(f"Prompt length: {len(prompt)}")
                logger.info(f"First 200 chars of prompt: {prompt[:200]}...")

                # Try direct OpenAI API first
                try:
                    if response_format == "json":
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=0,
                            presence_penalty=0,
                            response_format={"type": "json_object"}  # Changed from "json" to "json_object"
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                except APIError as e:
                    logger.warning(f"OpenAI API error: {e}. Falling back to proxy API.")
                    # Fall back to proxy API without response_format
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=0,
                        presence_penalty=0
                    )

                result = response.choices[0].message.content.strip()

                # Log response details
                logger.info("\nAPI Response Details:")
                logger.info(f"Response length: {len(result)}")
                logger.info(f"Response content: {result}")

                # Basic validation for JSON responses
                if response_format == "json":
                    # Basic validation
                    if not result:
                        logger.warning("Empty response received")
                        if attempt < max_retries - 1:
                            continue
                        return "[]"

                    # 验证JSON格式 - 允许对象或数组
                    is_array = result.startswith('[') and result.endswith(']')
                    is_object = result.startswith('{') and result.endswith('}')

                    if not (is_array or is_object):
                        logger.warning(f"Response is not a valid JSON format: {result}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            # Try to fix the response
                            fixed_result = self._fix_json_response(result)
                            if fixed_result:
                                return fixed_result
                            return "[]"

                    # 验证JSON解析
                    try:
                        parsed = json.loads(result)
                        # 接受数组或对象作为有效响应
                        if not (isinstance(parsed, list) or isinstance(parsed, dict)):
                            logger.warning("Parsed JSON is neither a list nor a dict")
                            if attempt < max_retries - 1:
                                continue
                            return "[]"
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed: {e}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            # Try to fix the response
                            fixed_result = self._fix_json_response(result)
                            if fixed_result:
                                return fixed_result
                            return "[]"
                else:
                    # For text responses, just return as is
                    return result

            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return "[]" if response_format == "json" else ""

    def _fix_json_response(self, response: str) -> Optional[str]:
        """
        Attempt to fix common JSON formatting issues in the response.

        Args:
            response: The problematic response string

        Returns:
            Fixed JSON string or None if unfixable
        """
        try:
            # 尝试直接解析，可能已经是有效的JSON
            try:
                json.loads(response)
                return response
            except:
                pass

            # 检测是否为JSON对象而非数组
            if '{' in response and '}' in response:
                if not (response.strip().startswith('[') and response.strip().endswith(']')):
                    object_start = response.find('{')
                    object_end = response.rfind('}')
                    if object_start >= 0 and object_end > object_start:
                        json_obj = response[object_start:object_end+1]

                        # 修复常见格式问题
                        json_obj = json_obj.replace("'", '"')        # 替换单引号
                        json_obj = json_obj.replace('\n', ' ')       # 移除换行符
                        json_obj = json_obj.replace('\t', ' ')       # 移除制表符
                        json_obj = re.sub(r',\s*}', '}', json_obj)  # 移除对象末尾的逗号

                        # 验证修复后的JSON对象
                        try:
                            json.loads(json_obj)
                            logger.info(f"Successfully fixed JSON object: {json_obj}")
                            return json_obj
                        except:
                            pass

            # 处理数组格式的JSON
            start = response.find('[')
            end = response.rfind(']')

            if start >= 0 and end > start:
                json_str = response[start:end+1]

                # 修复常见格式问题
                json_str = json_str.replace("'", '"')    # 替换单引号
                json_str = json_str.replace('\n', ' ')   # 移除换行符
                json_str = json_str.replace('\t', ' ')   # 移除制表符

                # 修复对象之间缺少逗号的问题
                json_str = json_str.replace('} {', '}, {')
                # 修复数组末尾的逗号
                json_str = json_str.replace(',]', ']')

                # 修复new_types格式中常见的问题
                if '"new_types"' in json_str:
                    # 修复examples数组中的格式问题
                    examples_pattern = r'"examples"\s*:\s*\[(.*?)\]'
                    for match in re.finditer(examples_pattern, json_str, re.DOTALL):
                        examples_content = match.group(1)
                        fixed_examples = re.sub(r',\s*\]', ']', examples_content)
                        json_str = json_str.replace(examples_content, fixed_examples)

                # 验证修复后的JSON
                try:
                    json.loads(json_str)
                    logger.info(f"Successfully fixed JSON array: {json_str}")
                    return json_str
                except json.JSONDecodeError as e:
                    logger.warning(f"Still invalid after initial fixes: {e}")

                    # 进一步尝试修复嵌套对象中的问题
                    try:
                        # 替换多余的逗号
                        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                        json.loads(json_str)
                        logger.info(f"Fixed nested object issues: {json_str}")
                        return json_str
                    except:
                        pass

            # 检查是否为单一值的情况
            single_value_pattern = r'"[^"]*"\s*:\s*"([^"]*)"'
            match = re.search(single_value_pattern, response)
            if match:
                value = match.group(1)
                logger.info(f"Extracted single value: {value}")
                # 返回一个简单的JSON对象
                return f'{{"value": "{value}"}}'

            return None

        except Exception as e:
            logger.error(f"Failed to fix JSON response: {e}")
            return None

    def parse_list_response(self, response: str) -> List[Any]:
        """
        Parse a list response from the LLM.

        Args:
            response: LLM response

        Returns:
            Parsed list data or dictionary with data
        """
        if not response or response.strip() == "[]" or response.strip() == "{}":
            return []

        try:
            # 尝试解析JSON字符串
            parsed = json.loads(response)

            # 记录解析结果的类型，帮助调试
            logger.info(f"Parsed JSON response type: {type(parsed).__name__}")

            # 检查特殊字段，处理非数组的响应格式
            if isinstance(parsed, dict):
                # 处理关系类型的特殊响应格式
                if "relationship_type" in parsed:
                    logger.info(f"Detected 'relationship_type' field: {parsed['relationship_type']}")
                    return parsed["relationship_type"]
                elif "relationship" in parsed:
                    logger.info(f"Detected 'relationship' field: {parsed['relationship']}")
                    return parsed["relationship"]
                # 处理实体类型的特殊响应格式
                elif "entity_type" in parsed:
                    logger.info(f"Detected 'entity_type' field: {parsed['entity_type']}")
                    return parsed["entity_type"]
                # 探索模块的新类型响应格式
                elif "new_types" in parsed:
                    logger.info(f"Detected 'new_types' field with {len(parsed['new_types'])} items")
                    return parsed  # 保持原样返回，探索模块期望这种格式

            # 返回解析后的对象，可以是列表或字典
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}. Attempting to fix response.")

            # 尝试修复常见的JSON格式问题
            fixed_response = self._fix_json_response(response)
            if fixed_response:
                try:
                    return json.loads(fixed_response)
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse fixed JSON: {e2}")

            # 尝试识别是否返回的是代码块中的JSON
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            code_match = re.search(code_block_pattern, response)
            if code_match:
                try:
                    json_str = code_match.group(1).strip()
                    logger.info(f"Extracted JSON from code block: {json_str[:100]}...")
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from code block")

            # 尝试从纯文本提取关系类型
            if "relationship_type" in response.lower() or "relationship" in response.lower():
                # 尝试捕获关系类型值
                relationship_pattern = r'["\'](is_a|part_of|equivalent_to|depends_on|prerequisite_for|related_to)["\']'
                match = re.search(relationship_pattern, response)
                if match:
                    logger.info(f"Extracted relationship type from text: {match.group(1)}")
                    return match.group(1)

            # 如果所有尝试都失败，返回空列表
            logger.error(f"Failed to parse response as JSON: {response[:200]}...")
            return []
