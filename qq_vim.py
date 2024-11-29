#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Tuple
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
import json
import os
import shutil
import sys
import urllib.request

HOME = os.environ["HOME"]
LOG_DIR = os.path.join(HOME, ".qq", "log")
CACHE_DIR = os.path.join(HOME, ".qq", "cache")
API_TOKENS_DIR = os.path.join(HOME, ".qq", "api_tokens")
CONF_PATH = os.path.join(HOME, ".qq", "conf")

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

ANTHROPIC_API_TOKEN = _load_api_token("ANTHROPIC", "anthropic.com")
DEEPSEEK_API_TOKEN  = _load_api_token("DEEPSEEK",  "deepseek.com")
TOGETHER_API_TOKEN  = _load_api_token("TOGETHER",  "together.ai")

def _load_conf():
    conf = ConfigParser()
    try:
        with open(CONF_PATH, "r") as conf_file:
            conf.read_file(conf_file)
    except Exception:
        conf = ConfigParser()
        conf["default"] = {
            "model": "qwq",
        }
        conf["aliases"] = {
            "llama": "llama-3.1-405b-instruct-quant8",
            "qwq": "qwen-qwq-32b-preview",
            "sonnet": "claude-3.5-sonnet-20241022",
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
    endpoint_max_tokens: int
    endpoint_api_url: str
    endpoint_api_token: str
    endpoint_protocol: str

    @classmethod
    def anthropic(cls, **kwargs) -> Any:
        return cls(
            endpoint_api_url = "https://api.anthropic.com",
            endpoint_api_token = ANTHROPIC_API_TOKEN,
            endpoint_protocol = "anthropic",
            **kwargs,
        )

    @classmethod
    def claude_3_5_sonnet_20241022(cls) -> Any:
        return cls.anthropic(
            model = "claude-3.5-sonnet-20241022",
            endpoint_model = "claude-3-5-sonnet-20241022",
            endpoint_max_tokens = 8192,
        )

    @classmethod
    def claude_3_5_sonnet_20240620(cls) -> Any:
        return cls.anthropic(
            model = "claude-3.5-sonnet-20240620",
            endpoint_model = "claude-3-5-sonnet-20240620",
            endpoint_max_tokens = 8192,
        )

    @classmethod
    def deepseek(cls, **kwargs) -> Any:
        return cls(
            # FIXME: beta API for 8192 max tokens.
            endpoint_api_url = "https://api.deepseek.com",
            #endpoint_api_url = "https://api.deepseek.com/beta",
            endpoint_api_token = DEEPSEEK_API_TOKEN,
            endpoint_protocol = "deepseek",
            **kwargs,
        )

    @classmethod
    def deepseek_v2_5_chat(cls) -> Any:
        return cls.deepseek(
            model = "deepseek-v2.5-chat",
            endpoint_model = "deepseek-chat",
            endpoint_max_tokens = 4096,
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
    def together_llama_3_1_405b_instruct_quant8(cls) -> Any:
        return cls.together(
            model = "llama-3.1-405b-instruct-quant8",
            endpoint_model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            endpoint_max_tokens = 4096,
        )

    @classmethod
    def together_qwen_qwq_32b_preview(cls) -> Any:
        return cls.together(
            model = "qwen-qwq-32b-preview",
            endpoint_model = "Qwen/QwQ-32B-Preview",
            endpoint_max_tokens = 16384,
            #endpoint_max_tokens = 32768,
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
        else:
            raise NotImplementedError

    def query(self, messages: List[Dict[str, str]], *args) -> Tuple[str, Any, Any]:
        if self.endpoint_protocol == "anthropic":
            if len(messages) > 0 and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]
            else:
                system_prompt = None
        if (
            self.endpoint_protocol == "anthropic" or
            self.endpoint_protocol == "deepseek" or
            self.endpoint_protocol == "openai"
        ):
            req_body = {
                "messages": messages,
                "model": self.endpoint_model,
                "stream": False,
                "max_tokens": self.endpoint_max_tokens,
                # TODO: configure sampling params.
                "temperature": 0,
                "top_p": 1,
            }
        else:
            raise NotImplementedError
        if (
            self.endpoint_protocol == "anthropic" and
            system_prompt is not None
        ):
            req_body["system"] = system_prompt
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
            response = res_body["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError
        return response, res_body, t1

@dataclass
class InferenceLog(InferenceEndpoint):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._base_dir = os.path.join(LOG_DIR, self.model)

    def query(self, messages: List[Dict[str, str]], src_path: str = None) -> Tuple[str, Any, Any]:
        t0 = datetime.utcnow()
        d = "{}".format(t0.date())
        t = "{}".format(t0.time())
        timestamp = "{}-{}".format(d, t[:8].replace(":", "_"))

        src_name = os.path.basename(src_path)

        log_name = "{}.{}.log.json".format(src_name, timestamp)
        log_path = os.path.join(self._base_dir, log_name)
        log_link = os.path.join(self._base_dir, "{}.latest.log.json".format(src_name))

        in_name = "{}.{}.in.txt".format(src_name, timestamp)
        in_path = os.path.join(self._base_dir, in_name)
        in_link = os.path.join(self._base_dir, "{}.latest.in.txt".format(src_name))

        out_name = "{}.{}.in.txt".format(src_name, timestamp)
        out_path = os.path.join(self._base_dir, out_name)
        out_link = os.path.join(self._base_dir, "{}.latest.out.txt".format(src_name))

        try:
            shutil.copyfile(src_path, in_path)
        except OSError:
            os.makedirs(LOG_DIR, exist_ok=True)
            shutil.copyfile(src_path, in_path)

        response, res_body, t1 = super().query(messages)

        log_item = {
            "t0": t0.isoformat(),
            "t1": t1.isoformat(),
            "messages": messages,
            "response": res_body,
        }
        with open(log_path, "w") as f:
            print(json.dumps(log_item, indent=2), file=f, flush=True)

        try:
            os.remove(log_link)
        except FileNotFoundError:
            pass
        try:
            os.remove(in_link)
        except FileNotFoundError:
            pass
        try:
            os.remove(out_link)
        except FileNotFoundError:
            pass

        os.symlink(log_name, log_link)

        if response.find(chr(0x11)) >= 0:
            print(f"DEBUG: found control char in response (0x11)")
            return
        if response.find(chr(0x01)) >= 0:
            print(f"DEBUG: found control char in response (0x01)")
            return

        with open(src_path, "a") as f:
            print(f"\n\n{AA_PAT}", file=f)
            print(response, file=f)
            if len(response) > 0 and response[-1] != "\n":
                print("", file=f)
            print(f"\n{QQ_PAT} ", file=f, flush=True)

        os.symlink(in_name, in_link)

        try:
            shutil.copyfile(src_path, out_path)
        except OSError:
            os.makedirs(LOG_DIR, exist_ok=True)
            shutil.copyfile(src_path, out_path)

        os.symlink(out_name, out_link)

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

    # TODO: system message.

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
            a_text = haystack[2:a_end].strip()
            print(f"DEBUG: message[{len(messages)}]: role = \"assistant\"")
            messages.append({
                "role": "assistant",
                "content": q_text,
            })
        if qq_pos < 0:
            break

        haystack = haystack[qq_pos+2:]
        aa_pos = haystack.find(AA_PAT)
        if aa_pos < 0:
            aa_pos = len(haystack)
        q_text = haystack[:aa_pos].strip()
        print(f"DEBUG: message[{len(messages)}]: role = \"user\"")
        messages.append({
            "role": "user",
            "content": q_text,
        })

        haystack = haystack[aa_pos:]

    if len(messages) <= 0:
        print(f"DEBUG: no messages")
        return

    if model == "llama-3.1-405b-instruct-quant8":
        endpoint = InferenceLog.together_llama_3_1_405b_instruct_quant8()
    elif model == "qwen-qwq-32b-preview":
        endpoint = InferenceLog.together_qwen_qwq_32b_preview()
    elif model == "claude-3.5-sonnet-20241022":
        endpoint = InferenceLog.claude_3_5_sonnet_20241022()
    else:
        print(f"DEBUG: unsupported model = {repr(model)}")
        return

    try:
        endpoint.query(messages, sys.argv[1])
    except Exception as e:
        print(f"DEBUG: endpoint query: except: {e}")
        return

if __name__ == "__main__":
    main()
