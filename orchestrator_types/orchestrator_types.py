from enum import Enum
from typing import List, Optional, Any
from dataclasses import dataclass
import time

class ConversationRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"

class ConversationMessage:
    role: ConversationRole
    original_user_input: str
    short_output: str
    tokens: Optional[int]
    content: List[Any]

    def __init__(self, role: ConversationRole, original_user_input: str, short_output: str, tokens: Optional[int], content: Optional[List[Any]] = None):
        self.role = role
        self.original_user_input = original_user_input
        self.short_output = short_output
        self.tokens = tokens
        self.content = content

class TimestampedMessage(ConversationMessage):
    def __init__(self,
                 role: ConversationRole,
                 original_user_input: str,
                 short_output: str,
                 content: Optional[List[Any]] = None,
                 tokens: Optional[int] = 0,
                 timestamp: Optional[int] = None):
        super().__init__(role, original_user_input, short_output, tokens, content)
        self.timestamp = timestamp or int(time.time() * 1000) 

@dataclass
class FinalResponse:
    AGENT_SELECTED: str
    AGENT_OUTPUT: str
    OUTPUT_TOKENS: int
    REQUEST_ID: str

@dataclass
class ExpectedResult:
    NUMBER_OF_AGENT_CALL: int
    AGENTS: List[str]
    RESULTS: List[str]

@dataclass
class OrchestratorConfig:
    AGENT_NOT_SELECTED_LOG: str = "I am sorry, I couldn't determine how to handle your request.\
    Could you please rephrase it?"
    ROUTING_ERROR_LOG: str = "I am Sorry, I couldn't determine the agent to handle your request"
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100 #required to limit message storage per agent and user_id, session_id