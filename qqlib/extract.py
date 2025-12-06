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
            for block in content:
                if block["type"] == "thinking":
                    pass
                elif block["type"] == "text":
                    if text is not None:
                        raise ValueError
                    text = block["text"]
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
            for block in content:
                if block["type"] == "thinking":
                    if thinking is not None:
                        raise ValueError
                    thinking = block["thinking"]
                elif block["type"] == "text":
                    pass
                else:
                    raise NotImplementedError
            return thinking
        else:
            raise NotImplementedError
