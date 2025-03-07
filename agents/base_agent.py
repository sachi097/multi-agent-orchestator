from typing import Dict, List, Union, AsyncIterable, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from orchestrator_types import ConversationMessage
import re

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
        original_user_input: str,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        pass

    def set_history(self, messages: List[ConversationMessage]) -> None:
        self.chat_history = self.format_messages(messages)

    @staticmethod
    def format_messages(messages: List[ConversationMessage]) -> str:
        return "\n".join([
            f"Original User Input: {message.original_user_input}\n" +
            f"{message.role}: {' '.join([message.content[0]['text']])}" for message in messages
        ])

    def update_system_prompt(self) -> None:
        all_variables: Dict[str, Union[str, List[str]]] = {
            "ORIGINAL_USER_INPUT": self.original_user_input,
            "SUBTASK_INPUT": self.subtask_input,
            "HISTORY": self.chat_history
        }
        self.system_prompt = self.replace_placeholders(self.prompt_template, all_variables)

    @staticmethod
    def replace_placeholders(template: str, variables: Dict[str, Union[str, List[str]]]) -> str:

        return re.sub(r'{{(\w+)}}',
                      lambda m: '\n'.join(variables.get(m.group(1), [m.group(0)]))
                      if isinstance(variables.get(m.group(1)), list)
                      else variables.get(m.group(1), m.group(0)), template)