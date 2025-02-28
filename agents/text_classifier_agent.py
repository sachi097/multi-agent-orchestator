from typing import Dict, List, Union, AsyncIterable, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from agents import Agent, AgentOptions
from orchestrator_types import (
    ConversationMessage,
    ConversationRole
)
from loguru import logger

OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"

@dataclass
class TextClassifierAgentOptions(AgentOptions):
    api_key: str = None
    model: Optional[str] = None
    streaming: Optional[bool] = None
    agent_config: Optional[Dict[str, Any]] = None
    client: Optional[Any] = None

class TextClassifierAgent(Agent):
    def __init__(self, options: TextClassifierAgentOptions):
        super().__init__(options)
        if not options.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = options.model or OPENAI_MODEL_ID_GPT_O_MINI
        self.streaming = options.streaming or False
        self.callbacks = options.callbacks
        
        if options.client:
            self.client = options.client
        else:
            self.client = OpenAI(api_key=options.api_key)

        # Default inference configuration
        default_agent_config = {
            'maxTokens': 1000,
            'temperature': None,
            'topP': None,
            'stopSequences': None
        }

        if options.agent_config:
            self.agent_config = {**default_agent_config, **options.agent_config}
        else:
            self.agent_config = default_agent_config

        # Initialize system prompt
        self.prompt_template = f"""You are a {self.name}.
        {self.description} Provide helpful and accurate information based on your expertise.
        You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise.
        The conversation will proceed as follows:
        - The human may ask an initial question or provide a prompt on any topic.
        - You will provide a relevant and informative response.
        - The human may then follow up with additional questions or prompts related to your previous response,
          allowing for a multi-turn dialogue on that topic.
        - Or, the human may switch to a completely new and unrelated topic at any point.
        - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses
          based on your broad knowledge base.
        Throughout the conversation, you should aim to:
        - Understand the context and intent behind each new question or prompt.
        - Provide substantive and well-reasoned responses that directly address the query.
        - Draw insights and connections from your extensive knowledge when appropriate.
        - Ask for clarification if any part of the question or prompt is ambiguous.
        - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style.
        - Seamlessly transition between topics as the human introduces new subjects."""

        self.system_prompt = ""
        
    def is_streaming_enabled(self) -> bool:
        return self.streaming is True

    async def handle_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        try:

            self.update_system_prompt()
            system_prompt = self.system_prompt

            messages = [
                {"role": "system", "content": system_prompt},
                *[{
                    "role": msg.role.lower(),
                    "content": msg.content[0].get('text', '') if msg.content else ''
                } for msg in chat_history],
                {"role": "user", "content": input_text}
            ]


            request_options = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.agent_config.get('maxTokens'),
                "temperature": self.agent_config.get('temperature'),
                "top_p": self.agent_config.get('topP'),
                "stop": self.agent_config.get('stopSequences'),
                "stream": self.streaming
            }
            if self.streaming:
                return await self.handle_streaming_response(request_options)
            else:
                return await self.handle_single_response(request_options)

        except Exception as error:
            logger.error(f"Error in OpenAI API call: {str(error)}")
            raise error

    async def handle_single_response(self, request_options: Dict[str, Any]) -> ConversationMessage:
        try:
            request_options['stream'] = False
            chat_completion = self.client.chat.completions.create(**request_options)

            if not chat_completion.choices:
                raise ValueError('No choices returned from OpenAI API')

            assistant_message = chat_completion.choices[0].message.content
            response_toknes = chat_completion.usage.completion_tokens

            if not isinstance(assistant_message, str):
                raise ValueError('Unexpected response format from OpenAI API')

            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                tokens=response_toknes,
                content=[{"text": assistant_message}]
            )

        except Exception as error:
            logger.error(f'Error in OpenAI API call: {str(error)}')
            raise error

    async def handle_streaming_response(self, request_options: Dict[str, Any]) -> ConversationMessage:
        try:
            stream = self.client.chat.completions.create(**request_options)
            accumulated_message = []
            tokens = 0
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    accumulated_message.append(chunk_content)
                    tokens = tokens + len(chunk_content.split(' '))
                    if self.callbacks:
                        self.callbacks.on_llm_new_token(chunk_content)

            # Store the complete message in the instance for later access if needed
            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                tokens=tokens,
                content=[{"text": ''.join(accumulated_message)}]
            )

        except Exception as error:
            logger.error(f"Error getting stream from OpenAI model: {str(error)}")
            raise error

    def update_system_prompt(self) -> None:
        self.system_prompt = self.prompt_template