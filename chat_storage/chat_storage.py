from abc import ABC, abstractmethod
from typing import Optional, Union
from orchestrator_types import ConversationMessage, TimestampedMessage

class ChatStorage(ABC):
    """Abstract base class representing the interface for chat storage."""
    def is_same_role_as_last_message(self,
                               conversation: list[ConversationMessage],
                               new_message: ConversationMessage) -> bool:
        """
        Check if the new message is consecutive with the last message in the conversation.
        Returns:
            bool: True if the new message is consecutive, False otherwise.
        """
        if not conversation:
            return False
        return conversation[-1].role == new_message.role

    def trim_conversation(self,
                          conversation: list[ConversationMessage],
                          max_history_size: Optional[int] = None) -> list[ConversationMessage]:
        """
        Trim the conversation to the specified maximum history size.
        Returns:
            list[ConversationMessage]: The trimmed conversation.
        """
        if max_history_size is None:
            return conversation
        # Ensure max_history_size is even to maintain complete binoms
        if max_history_size % 2 == 0:
            adjusted_max_history_size = max_history_size
        else:
            adjusted_max_history_size = max_history_size - 1
        return conversation[-adjusted_max_history_size:]

    @abstractmethod
    async def save_message(self,
                                user_id: str,
                                session_id: str,
                                agent_id: str,
                                new_message: Union[ConversationMessage, TimestampedMessage],
                                max_history_size: Optional[int] = None) -> bool:
        """
        Save a new chat message.
        Returns:
            bool: True if the message was saved successfully, False otherwise.
        """

    @abstractmethod
    async def save_messages(self,
                                user_id: str,
                                session_id: str,
                                agent_id: str,
                                new_messages: Union[list[ConversationMessage], list[TimestampedMessage]],
                                max_history_size: Optional[int] = None) -> bool:
        """
        Save multiple messages at once.
        Returns:
            bool: True if the messages were saved successfully, False otherwise.
        """

    @abstractmethod
    async def fetch_chat(self,
                         user_id: str,
                         session_id: str,
                         agent_id: str,
                         max_history_size: Optional[int] = None) -> list[ConversationMessage]:
        """
        Fetch chat messages.
        Returns:
            list[ConversationMessage]: The fetched chat messages.
        """

    @abstractmethod
    async def fetch_all_chats(self,
                              user_id: str,
                              session_id: str) -> list[ConversationMessage]:
        """
        Fetch all chat messages for a user and session.
        Returns:
            list[ConversationMessage]: All chat messages for the user and session.
        """