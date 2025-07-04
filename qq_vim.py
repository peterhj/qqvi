#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Tuple
from argparse import Namespace
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
import json
import os
import shutil
import sys
import traceback
import urllib.request

HOME = os.environ["HOME"]
LOG_DIR = os.path.join(HOME, ".qq", "log")
API_TOKENS_DIR = os.path.join(HOME, ".qq", "api_tokens")
CONF_PATH = os.path.join(HOME, ".qq", "conf")

_EXTRA_LOG_DIR = os.environ.get("QQ_EXTRA_LOG_DIR", None)

def _load_api_token(key, domain):
    env_key = "{}_API_KEY".format(key)
    name = "{}.txt".format(domain)
    path = os.path.join(API_TOKENS_DIR, name)
    api_token = os.environ.get(env_key)
    if api_token is None:
        try:
            with open(path, "r") as api_token_file:
                api_token = api_token_file.read().strip()
        except:
            pass
    return api_token

DEEPINFRA_API_TOKEN = _load_api_token("DEEPINFRA",  "deepinfra.com")
TOGETHER_API_TOKEN  = _load_api_token("TOGETHER",   "together.ai")
HYPERBOLIC_API_TOKEN = _load_api_token("HYPERBOLIC", "hyperbolic.xyz")
OPENROUTER_API_TOKEN = _load_api_token("OPENROUTER", "openrouter.ai")
ANTHROPIC_API_TOKEN = _load_api_token("ANTHROPIC",  "anthropic.com")
DEEPSEEK_API_TOKEN  = _load_api_token("DEEPSEEK",   "deepseek.com")
GEMINI_API_TOKEN    = _load_api_token("GEMINI",     "aistudio.google.com")
OPENAI_API_TOKEN    = _load_api_token("OPENAI",     "openai.com")
XAI_API_TOKEN       = _load_api_token("XAI",        "x.ai")

def _load_conf():
    conf = ConfigParser()
    try:
        with open(CONF_PATH, "r") as conf_file:
            conf.read_file(conf_file)
    except Exception:
        conf = ConfigParser()
        conf["default"] = {
            "model": "deepseek-r1",
        }
        conf["aliases"] = {
            "claude": "claude-3.7-sonnet-20250219",
            "deepseek": "deepseek-v3-chat-20250324",
            "deepseek-r1": "deepseek-r1-20250528",
            "llama": "llama-3.1-405b-instruct-quant8",
            "sonnet": "claude-3.7-sonnet-20250219",
        }
        try:
            with open(CONF_PATH, "w") as conf_file:
                conf.write(conf_file)
        except Exception:
            pass
    return conf

CONF = _load_conf()

QQ_PAT = f"{chr(0x11)}{chr(0x11)}"
AA_PAT = f"{chr(0x01)}{chr(0x01)}"

@dataclass
class InferenceEndpoint:
    model: Optional[str]
    endpoint_model: str
    endpoint_max_new_tokens: int
    endpoint_api_url: str
    endpoint_api_token: str
    endpoint_protocol: str
    endpoint_extra_params: Optional[dict] = None

    @classmethod
    def deepinfra(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.deepinfra.com",
            endpoint_api_token = DEEPINFRA_API_TOKEN,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def claude_4_sonnet_20250514_deepinfra(cls) -> Any:
        return cls.deepinfra(
            model = "claude-4-sonnet-20250514",
            endpoint_model = "anthropic/claude-4-sonnet",
            endpoint_max_new_tokens = 100000,
        )

    @classmethod
    def claude_4_opus_20250514_deepinfra(cls) -> Any:
        return cls.deepinfra(
            model = "claude-4-opus-20250514",
            endpoint_model = "anthropic/claude-4-opus",
            endpoint_max_new_tokens = 100000,
        )

    @classmethod
    def gemini_2_5_flash_20250617_deepinfra(cls) -> Any:
        return cls.deepinfra(
            model = "gemini-2.5-flash-20250617",
            endpoint_model = "google/gemini-2.5-flash",
            endpoint_max_new_tokens = 131072,
        )

    @classmethod
    def gemini_2_5_pro_20250617_deepinfra(cls) -> Any:
        return cls.deepinfra(
            model = "gemini-2.5-pro-20250617",
            endpoint_model = "google/gemini-2.5-pro",
            endpoint_max_new_tokens = 131072,
        )

    @classmethod
    def together(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.together.xyz",
            endpoint_api_token = TOGETHER_API_TOKEN,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def together_deepseek_v3(cls) -> Any:
        return cls.together(
            model = "deepseek-v3",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def together_llama_3_1_405b_instruct_quant8(cls) -> Any:
        return cls.together(
            model = "llama-3.1-405b-instruct-quant8",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def together_qwen_qwq_32b_preview(cls) -> Any:
        return cls.together(
            model = "qwq-32b-preview",
            endpoint_model = "Qwen/QwQ-32B-Preview",
            endpoint_max_new_tokens = 16384,
            #endpoint_max_new_tokens = 32768,
        )

    @classmethod
    def hyperbolic(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.hyperbolic.xyz",
            endpoint_api_token = HYPERBOLIC_API_TOKEN,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def hyperbolic_deepseek_r1_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-deepseek-r1-20250120",
            endpoint_model = "deepseek-ai/DeepSeek-R1",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_r1_zero_20250120(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-deepseek-r1-zero-20250120",
            endpoint_model = "deepseek-ai/DeepSeek-R1-Zero",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_v3_20250324(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-deepseek-v3-20250324",
            endpoint_model = "deepseek-ai/DeepSeek-V3-0324",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_deepseek_v3_20241226(cls) -> Any:
        return cls.hyperbolic(
            model = "hyperbolic-deepseek-v3-20241226",
            endpoint_model = "deepseek-ai/DeepSeek-V3",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_bf16(cls) -> Any:
        return cls.hyperbolic(
            model = "llama-3.1-405b-base",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def hyperbolic_llama_3_1_405b_base_fp8(cls) -> Any:
        return cls.hyperbolic(
            model = "llama-3.1-405b-base-fp8",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-FP8",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def anthropic(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.anthropic.com",
            endpoint_api_token = ANTHROPIC_API_TOKEN,
            endpoint_protocol = "anthropic",
            **kwargs,
        )

    @classmethod
    def claude_3_7_sonnet_20250219(cls) -> Any:
        return cls.anthropic(
            model = "claude-3.7-sonnet-20250219",
            endpoint_model = "claude-3-7-sonnet-20250219",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def claude_3_5_sonnet_20241022(cls) -> Any:
        return cls.anthropic(
            model = "claude-3.5-sonnet-20241022",
            endpoint_model = "claude-3-5-sonnet-20241022",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def claude_3_5_sonnet_20240620(cls) -> Any:
        return cls.anthropic(
            model = "claude-3.5-sonnet-20240620",
            endpoint_model = "claude-3-5-sonnet-20240620",
            endpoint_max_new_tokens = 8192,
        )

    @classmethod
    def deepseek(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.deepseek.com",
            endpoint_api_token = DEEPSEEK_API_TOKEN,
            endpoint_protocol = "deepseek",
            **kwargs,
        )

    @classmethod
    def deepseek_r1_20250528(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-r1-20250528",
            endpoint_model = "deepseek-reasoner",
            endpoint_max_new_tokens = 8192,
            #endpoint_max_context_len = 65536,
        )

    @classmethod
    def deepseek_r1_20250120(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-r1-20250120",
            endpoint_model = "deepseek-reasoner",
            endpoint_max_new_tokens = 8192,
            #endpoint_max_context_len = 65536,
        )

    @classmethod
    def deepseek_v3_chat_20250324(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v3-chat-20250324",
            endpoint_model = "deepseek-chat",
            endpoint_max_new_tokens = 8192,
            #endpoint_max_context_len = 65536,
        )

    @classmethod
    def deepseek_v3_chat_20241226(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v3-chat-20241226",
            endpoint_model = "deepseek-chat",
            endpoint_max_new_tokens = 8192,
            #endpoint_max_context_len = 65536,
        )

    @classmethod
    def deepseek_v2_5_chat_20241210(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v2.5-chat-20241210",
            endpoint_model = "deepseek-chat",
            endpoint_max_new_tokens = 4096,
        )

    @classmethod
    def openai(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.openai.com",
            endpoint_api_token = OPENAI_API_TOKEN,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def o3_20250416(cls) -> Any:
        return cls.openai(
            model = "o3-20250416",
            endpoint_model = "o3",
            endpoint_max_new_tokens = 100000,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
        )

    @classmethod
    def o4_mini_20250416(cls) -> Any:
        return cls.openai(
            model = "o4-mini-20250416",
            endpoint_model = "o4-mini",
            endpoint_max_new_tokens = 100000,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
        )

    @classmethod
    def xai(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.x.ai",
            endpoint_api_token = XAI_API_TOKEN,
            endpoint_protocol = "openai",
            **kwargs,
        )

    @classmethod
    def grok_3_mini(cls) -> Any:
        return cls.xai(
            model = "grok-3-mini-20250520",
            endpoint_model = "grok-3-mini",
            endpoint_max_new_tokens = 131072,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
        )

    @classmethod
    def grok_3_mini_beta(cls) -> Any:
        return cls.xai(
            model = "grok-3-mini-beta-20250418",
            endpoint_model = "grok-3-mini-beta",
            #endpoint_max_new_tokens = 32768,
            #endpoint_max_new_tokens = 65536,
            endpoint_max_new_tokens = 131072,
            endpoint_extra_params = {
                "reasoning_effort": "high",
            },
        )

    @classmethod
    def grok_3(cls) -> Any:
        return cls.xai(
            model = "grok-3-beta-20250520",
            endpoint_model = "grok-3",
            endpoint_max_new_tokens = 131072,
        )

    @classmethod
    def grok_3_beta(cls) -> Any:
        return cls.xai(
            model = "grok-3-beta-20250418",
            endpoint_model = "grok-3-beta",
            #endpoint_max_new_tokens = 32768,
            #endpoint_max_new_tokens = 65536,
            endpoint_max_new_tokens = 131072,
        )

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.endpoint_model
        if self.endpoint_protocol == "anthropic":
            # TODO: proper urllib formatting.
            self._chat_endpoint_url = "{}/v1/messages".format(self.endpoint_api_url)
            #print("DEBUG: chat endpoint url = {}".format(self._chat_endpoint_url))
            self._chat_endpoint_headers = {
                "User-Agent": "curl/8.7.1",
                "X-API-Key": "{}".format(self.endpoint_api_token),
                "Anthropic-Version": "2023-06-01",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            #print("DEBUG: chat endpoint headers = {}".format(self._chat_endpoint_headers))
        elif (
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            # TODO: proper urllib formatting.
            if self.endpoint_protocol == "deepseek":
                self._chat_endpoint_url = "{}/chat/completions".format(self.endpoint_api_url)
            elif self.endpoint_protocol == "openai":
                self._chat_endpoint_url = "{}/v1/chat/completions".format(self.endpoint_api_url)
            else:
                raise NotImplementedError
            self._chat_endpoint_headers = {
                "User-Agent": "curl/8.7.1",
                "Authorization": "Bearer {}".format(self.endpoint_api_token),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        elif self.endpoint_protocol == "gemini":
            self._chat_endpoint_url = "{}/v1beta/models/{}:generateContent?key={}".format(
                self.endpoint_api_url,
                self.endpoint_model,
                self.endpoint_api_token,
            )
            self._chat_endpoint_headers = {
                "User-Agent": "curl/8.7.1",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        else:
            raise NotImplementedError

    def query(self, messages: List[Dict[str, str]], *args) -> Tuple[str, Any, Any]:
        if self.endpoint_protocol == "anthropic":
            if len(messages) > 0 and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
            else:
                system_prompt = None
        if self.model == "qwq-32b-preview":
            messages.append(
                {
                    "role": "assistant",
                    #"content": "Let's plan step-by-step and review the steps.",
                    "content": "Let's plan our steps and review the steps.",
                }
            )
        if (
            self.endpoint_protocol == "anthropic" or
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            req_body = {
                "messages": messages,
                "model": self.endpoint_model,
                "stream": False,
            }
            if (
                self.endpoint_protocol == "deepseek" and
                self.endpoint_model == "deepseek-reasoner"
            ):
                req_body["max_new_tokens"] = self.endpoint_max_new_tokens
            elif (
                self.endpoint_protocol == "deepseek"
            ):
                req_body |= {
                    "max_new_tokens": self.endpoint_max_new_tokens,
                    # TODO: configure sampling params.
                    "temperature": 0,
                    #"top_p": 1,
                }
            elif (
                self.endpoint_protocol == "openai" and (
                    self.endpoint_model == "o3" or
                    self.endpoint_model.startswith("o3-") or
                    self.endpoint_model.startswith("o4-")
                )
            ):
                req_body["max_completion_tokens"] = self.endpoint_max_new_tokens
            else:
                req_body |= {
                    # TODO: max_tokens is input + output.
                    "max_tokens": self.endpoint_max_new_tokens,
                    # TODO: configure sampling params.
                    "temperature": 0,
                    "top_p": 1,
                    "logprobs": True,
                }
        elif self.endpoint_protocol == "gemini":
            # TODO
            req_body = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": messages[-1]["content"],
                            }
                        ],
                    }
                ],
            }
        else:
            raise NotImplementedError
        if (
            self.endpoint_protocol == "anthropic" and
            system_prompt is not None
        ):
            req_body["system"] = system_prompt
        if self.endpoint_extra_params is not None:
            req_body |= self.endpoint_extra_params
        req_data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            self._chat_endpoint_url,
            headers = self._chat_endpoint_headers.copy(),
            data = req_data,
        )
        with urllib.request.urlopen(req) as res:
            res_data = res.read()
        t1 = datetime.utcnow()
        res_body = json.loads(res_data.decode("utf-8"))
        if self.endpoint_protocol == "anthropic":
            response = res_body["content"][0]["text"]
        elif (
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            if "reasoning_content" in res_body["choices"][0]["message"]:
                response = "<think>\n{}\n</think>\n\n{}".format(
                    res_body["choices"][0]["message"]["reasoning_content"],
                    res_body["choices"][0]["message"]["content"],
                )
            else:
                response = res_body["choices"][0]["message"]["content"]
        elif self.endpoint_protocol == "gemini":
            response = res_body["candidates"][0]["parts"][-1]["text"]
        else:
            raise NotImplementedError
        return response, res_body, t1

@dataclass
class InferenceLog(InferenceEndpoint):
    def __post_init__(self) -> None:
        super().__post_init__()

    def query(self, messages: List[Dict[str, str]], src_path: str = None) -> Tuple[str, Any, Any]:
        t0 = datetime.utcnow()
        d = "{}".format(t0.date())
        t = "{}".format(t0.time())
        timestamp = "{}-{}".format(d, t[:8].replace(":", "_"))

        src_name = os.path.basename(src_path)

        log_dirs = [LOG_DIR]
        if _EXTRA_LOG_DIR is not None:
            log_dirs.append(_EXTRA_LOG_DIR)

        log_meta = dict()

        for log_dir in log_dirs:
            log_meta[log_dir] = Namespace(**{
                "base_dir": None,
                "log_name": None,
                "log_path": None,
                "log_link": None,
                "in_name": None,
                "in_path": None,
                "in_link": None,
                "out_name": None,
                "out_path": None,
                "out_link": None,
            })

            base_dir = os.path.join(log_dir, self.model)

            log_meta[log_dir].log_name = "{}.{}.log.json".format(src_name, timestamp)
            log_meta[log_dir].log_path = os.path.join(base_dir, log_meta[log_dir].log_name)
            log_meta[log_dir].log_link = os.path.join(base_dir, "{}.latest.log.json".format(src_name))

            log_meta[log_dir].in_name = "{}.{}.in.txt".format(src_name, timestamp)
            log_meta[log_dir].in_path = os.path.join(base_dir, log_meta[log_dir].in_name)
            log_meta[log_dir].in_link = os.path.join(base_dir, "{}.latest.in.txt".format(src_name))

            log_meta[log_dir].out_name = "{}.{}.out.txt".format(src_name, timestamp)
            log_meta[log_dir].out_path = os.path.join(base_dir, log_meta[log_dir].out_name)
            log_meta[log_dir].out_link = os.path.join(base_dir, "{}.latest.out.txt".format(src_name))

            try:
                shutil.copyfile(src_path, log_meta[log_dir].in_path)
            except OSError:
                os.makedirs(base_dir, exist_ok=True)
                shutil.copyfile(src_path, log_meta[log_dir].in_path)

        response, res_body, t1 = super().query(messages)

        log_item = {
            "t0": t0.isoformat(),
            "t1": t1.isoformat(),
            "messages": messages,
            "response": res_body,
        }

        for log_dir in log_dirs:
            with open(log_meta[log_dir].log_path, "w") as f:
                print(json.dumps(log_item, indent=2), file=f, flush=True)

            try:
                os.remove(log_meta[log_dir].log_link)
            except FileNotFoundError:
                pass
            try:
                os.remove(log_meta[log_dir].in_link)
            except FileNotFoundError:
                pass
            try:
                os.remove(log_meta[log_dir].out_link)
            except FileNotFoundError:
                pass

            os.symlink(log_meta[log_dir].log_name, log_meta[log_dir].log_link)

        if response.find(chr(0x11)) >= 0:
            print(f"DEBUG: found control char in response (0x11)")
            return
        if response.find(chr(0x01)) >= 0:
            print(f"DEBUG: found control char in response (0x01)")
            return

        with open(src_path, "a") as f:
            print(f"\n\n{AA_PAT}", file=f)
            if messages[-1]["role"] == "assistant":
                if "reasoning_content" in messages[-1]:
                    print("<think>\n{}\n</think>\n".format(messages[-1]["reasoning_content"]), end="", file=f)
                print(messages[-1]["content"], end="", file=f)
            print(response, file=f)
            if len(response) > 0 and response[-1] != "\n":
                print("", file=f)
            print(f"\n{QQ_PAT} ", file=f, flush=True)

        for log_dir in log_dirs:
            os.symlink(log_meta[log_dir].in_name, log_meta[log_dir].in_link)

            try:
                shutil.copyfile(src_path, log_meta[log_dir].out_path)
            except OSError:
                os.makedirs(log_dir, exist_ok=True)
                shutil.copyfile(src_path, log_meta[log_dir].out_path)

            os.symlink(log_meta[log_dir].out_name, log_meta[log_dir].out_link)

        return response, res_body, t1

def main():
    if len(sys.argv) <= 1:
        print(f"DEBUG: no src")
        return

    if len(sys.argv) <= 2:
        if "default" not in CONF:
            print(f"DEBUG: no default")
            return
        if "model" not in CONF["default"]:
            print(f"DEBUG: no default model")
            return
        model = CONF["default"]["model"]
    else:
        model = sys.argv[2]

    if (
        "aliases" in CONF and
        model in CONF["aliases"]
    ):
        model = CONF["aliases"][model]

    with open(sys.argv[1], "r") as f:
        text = f.read()

    haystack = text
    aa_pos = -1
    messages = []

    while len(haystack) > 0:
        qq_pos = haystack.find(QQ_PAT)
        if aa_pos < 0 and qq_pos > 0:
            y_text = haystack[:qq_pos].strip()
            if len(y_text) > 0:
                print(f"DEBUG: message[{len(messages)}]: role = \"system\"")
                messages.append({
                    "role": "system",
                    "content": y_text,
                })
        elif aa_pos >= 0:
            if qq_pos < 0:
                a_end = len(haystack)
            else:
                a_end = qq_pos
            a_text = haystack[:a_end].strip()
            print(f"DEBUG: message[{len(messages)}]: role = \"assistant\"")
            messages.append({
                "role": "assistant",
                "content": a_text,
            })
        if qq_pos < 0:
            break
        haystack = haystack[qq_pos+2:]

        aa_pos = haystack.find(AA_PAT)
        if aa_pos < 0:
            q_end = len(haystack)
        else:
            q_end = aa_pos
        q_text = haystack[:q_end].strip()
        print(f"DEBUG: message[{len(messages)}]: role = \"user\"")
        messages.append({
            "role": "user",
            "content": q_text,
        })
        if aa_pos < 0:
            break
        haystack = haystack[aa_pos+2:]

    if len(messages) <= 0:
        print(f"DEBUG: no messages")
        return

    if model == "llama-3.1-405b-instruct-quant8":
        endpoint = InferenceLog.together_llama_3_1_405b_instruct_quant8()
    elif model == "claude-4-sonnet-20250514":
        endpoint = InferenceLog.claude_4_sonnet_20250514_deepinfra()
    elif model == "claude-4-opus-20250514":
        endpoint = InferenceLog.claude_4_opus_20250514_deepinfra()
    elif model == "claude-3.5-sonnet-20241022":
        endpoint = InferenceLog.claude_3_5_sonnet_20241022()
    elif model == "deepseek-v3-chat-20250324":
        endpoint = InferenceLog.deepseek_v3_chat_20250324()
    elif model == "deepseek-r1-20250528":
        endpoint = InferenceLog.deepseek_r1_20250528()
    elif model == "gemini-2.5-flash-20250617":
        endpoint = InferenceLog.gemini_2_5_flash_20250617_deepinfra()
    elif model == "gemini-2.5-pro-20250617":
        endpoint = InferenceLog.gemini_2_5_pro_20250617_deepinfra()
    elif model in (
        "grok-3-mini-beta",
        "grok-3-mini-beta-20250418",
    ):
        endpoint = InferenceLog.grok_3_mini_beta()
    elif model in (
        "grok-3-beta",
        "grok-3-beta-20250418",
    ):
        endpoint = InferenceLog.grok_3_beta()
    elif model in ("o3", "o3-20250416"):
        endpoint = InferenceLog.o3_20250416()
    elif model in ("o4-mini", "o4-mini-20250416"):
        endpoint = InferenceLog.o4_mini_20250416()
    else:
        print(f"DEBUG: unsupported model = {repr(model)}")
        return

    try:
        endpoint.query(messages, sys.argv[1])
    except Exception as e:
        print(f"DEBUG: endpoint query: except: {e}")
        print(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
