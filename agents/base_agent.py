from typing import Dict, List, Union, AsyncIterable, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from orchestrator_types import ConversationMessage

@dataclass
class AgentResponse:
    output: Union[ConversationMessage, str]
    streaming: bool

class AgentCallbacks:
    def on_llm_new_token(self, token: str) -> None:
        pass

@dataclass
class AgentOptions:
    name: str
    description: str
    save_chat: bool = True
    streaming: bool = False
    callbacks: Optional[AgentCallbacks] = None

class Agent(ABC):
    def __init__(self, options: AgentOptions):
        self.name = options.name
        self.id = self.generate_key_from_name(options.name)
        self.description = options.description
        self.save_chat = options.save_chat

    def is_streaming_enabled(self) -> bool:
        return False

    @staticmethod
    def generate_key_from_name(name: str) -> str:
        import re

        # Remove special characters and replace spaces with hyphens
        key = re.sub(r"[^a-zA-Z0-9\s-]", "", name)
        key = re.sub(r"\s+", "-", key)
        return key.lower()

    @abstractmethod
    async def handle_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        pass