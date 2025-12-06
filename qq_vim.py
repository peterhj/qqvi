#!/usr/bin/env python3

from typing import Any, Optional
from argparse import Namespace
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
import os
import shutil
import sys
import traceback

from qqlib.api import APIServices
from qqlib.extract import Message

HOME = os.environ["HOME"]
LOG_DIR = os.path.join(HOME, ".qq", "log")
CONF_PATH = os.path.join(HOME, ".qq", "conf")

def _load_conf():
    conf = ConfigParser()
    try:
        with open(CONF_PATH, "r") as conf_file:
            conf.read_file(conf_file)
    except Exception:
        conf = ConfigParser()
        conf["default"] = {
            "model": "deepseek-ai/deepseek-v3.2-thinking",
            # "model": "deepseek-ai/deepseek-v3.2-speciale-thinking",
            # "model": "anthropic/claude-4.5-sonnet-thinking-32k",
            # "model": "x-ai/grok-4.1-fast-thinking",
            # "model": "__local__/openai/gpt-oss-20b",
            "max_tokens": 65536,
            "temperature": 1.0,
        }
        conf["aliases"] = {
            # "grok-fast": "x-ai/grok-4.1-fast-thinking",
        }
        try:
            with open(CONF_PATH, "w") as conf_file:
                conf.write(conf_file)
        except Exception:
            pass
    conf = _convert_conf_to_dict(conf)
    return conf

def _convert_conf_to_dict(conf):
    root = isinstance(conf, ConfigParser)
    section = isinstance(conf, SectionProxy)
    if root or section:
        d = dict()
        for k, v in conf.items():
            v2 = _convert_conf_to_dict(v)
            if root and k == "DEFAULT":
                d["__default__"] = v2
            else:
                d[k] = v2
        return d
    else:
        return conf

def get_conf(full_key: str, type, default=None):
    keys = full_key.split(".")
    conf = CONF
    value = None
    for level, k in enumerate(keys):
        assert k
        value = conf.get(k, None)
        if value is None:
            return default
        conf = value
    assert value is not None
    value = type(value)
    return value

CONF = _load_conf()

QQ_PAT = f"{chr(0x11)}{chr(0x11)}"
AA_PAT = f"{chr(0x01)}{chr(0x01)}"
V_PAT  = f"{chr(0x16)}"

@dataclass
class InferenceLog:
    services: APIServices
    model: Any

    def __post_init__(self) -> None:
        pass

    async def query(self, messages: list[dict[str, str]], src_path: str = None) -> tuple[str, Any, Any]:
        t0 = datetime.utcnow()
        d = "{}".format(t0.date())
        t = "{}".format(t0.time())
        timestamp = "{}-{}".format(d, t[:8].replace(":", "_"))

        src_name = os.path.basename(src_path)

        log_dirs = [LOG_DIR]

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

            base_dir = os.path.join(log_dir, self.model.model_path.lower())

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

        sampling_params = self.model.default_sampling_params
        if sampling_params is None:
            # default_max_tokens = 32768
            # default_temperature = 0.6
            default_max_tokens = 65536
            default_temperature = 1.0
            default_top_p = 1.0
            default_top_k = -1
            sampling_params = {
                "max_tokens": get_conf("default.max_tokens", int, default_max_tokens),
                "temperature": get_conf("default.temperature", float, default_temperature),
                "top_p": get_conf("default.top_p", float, default_top_p),
                "top_k": get_conf("default.top_k", int, default_top_k),
            }
        res = await self.services.client.message(
            self.model,
            messages,
            sampling_params,
            fresh=True,
        )
        t1 = datetime.utcnow()

        res_body = res.result()
        res_message = res.message()
        thinking = Message.get_thinking(res_message)
        response = Message.get_text(res_message)

        log_item = {
            "t0": t0.isoformat(),
            "t1": t1.isoformat(),
            "messages": messages,
            "sampling_params": sampling_params,
            "result": res_body,
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
            if thinking:
                print("<think>", file=f)
                thinking_end = len(thinking)
                if len(thinking) >= 2 and thinking[-2:] == "\n\n":
                    thinking_end = thinking_end - 2
                elif len(thinking) >= 1 and thinking[-1:] == "\n":
                    thinking_end = thinking_end - 1
                print(thinking[:thinking_end], file=f)
                print("</think>\n", file=f)
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

    services = APIServices()
    model_path = model
    model = services.registry.find_model(model_path)
    if model is None:
        print(f"DEBUG: unsupported model = {repr(model_path)}")
        return

    with open(sys.argv[1], "r") as f:
        haystack = f.read()

    y_text = None
    qq_pos = haystack.find(QQ_PAT)
    aa_pos = -1
    messages = []

    if qq_pos > 0:
        y_text = haystack[:qq_pos].strip()
    if y_text:
        print(f"DEBUG: message[{len(messages)}]: role = \"system\"")
        messages.append({
            "role": "system",
            "content": y_text,
        })

    while len(haystack) > 0:
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
        qq_pos = haystack.find(QQ_PAT)

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

    if len(messages) <= 0:
        print(f"DEBUG: no messages")
        return

    async def _run(services, model):
        endpoint = InferenceLog(services, model)
        try:
            await endpoint.query(messages, sys.argv[1])
        except Exception as e:
            print(f"DEBUG: endpoint query: except: {e}")
            print(traceback.format_exc())
    asyncio.run(_run(services, model))

if __name__ == "__main__":
    main()
