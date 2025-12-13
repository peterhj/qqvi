from typing import Optional, TypedDict, Union

class MessagePart(TypedDict):
    type: str
    text: Optional[str]
    thinking: Optional[str]
    signature: Optional[str]

class Message(TypedDict):
    role: str
    content: Union[str, list[MessagePart]]

    @staticmethod
    def get_text(message: dict) -> Optional[str]:
        content = message.get("content", None)
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text = None
            for part in content:
                if part["type"] == "thinking":
                    pass
                elif part["type"] == "text":
                    if text is not None:
                        raise ValueError
                    text = part["text"]
                else:
                    raise NotImplementedError
            return text
        else:
            raise NotImplementedError

    @staticmethod
    def get_thinking(message: dict) -> Optional[str]:
        content = message.get("content", None)
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            thinking = None
            for part in content:
                if part["type"] == "thinking":
                    if thinking is not None:
                        raise ValueError
                    thinking = part["thinking"]
                elif part["type"] == "text":
                    pass
                else:
                    raise NotImplementedError
            return thinking
        else:
            raise NotImplementedError

    @staticmethod
    def get_thinking_part(message: dict) -> Optional[dict]:
        content = message.get("content", None)
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            thinking_part = None
            for part in content:
                if part["type"] == "thinking":
                    if thinking_part is not None:
                        raise ValueError
                    thinking_part = part
                elif part["type"] == "text":
                    pass
                else:
                    raise NotImplementedError
            return thinking_part
        else:
            raise NotImplementedError
