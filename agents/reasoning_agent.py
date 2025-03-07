from typing import Dict, List, Union, AsyncIterable, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from agents import Agent, AgentOptions
from orchestrator_types import (
    ConversationMessage,
    ConversationRole
)
from loguru import logger
import json

OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"

@dataclass
class ReasoningAgentOptions(AgentOptions):
    api_key: str = None
    model: Optional[str] = None
    streaming: Optional[bool] = None
    agent_config: Optional[Dict[str, Any]] = None
    client: Optional[Any] = None



class ReasoningAgent(Agent):
    def __init__(self, options: ReasoningAgentOptions):
        super().__init__(options)
        if not options.api_key:
            raise ValueError("OpenAI API key is required")
        
        if options.client:
            self.client = options.client
        else:
            self.client = OpenAI(api_key=options.api_key)
        self.callbacks = options.callbacks
                
        self.model = options.model or OPENAI_MODEL_ID_GPT_O_MINI
        self.streaming = options.streaming or False

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
        You specialize only in tasks to {self.description} 
        Provide helpful and accurate information based on your expertise.
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
        - Seamlessly transition between topics as the human introduces new subjects.""" + """
        ###Guidelines###
        - Original user input is 
        <original_user_input>
         {{ORIGINAL_USER_INPUT}}
        </orignal_user_input>

        - Current sub-task input is
        <subtask_input>
        {{SUBTASK_INPUT}}
        </subtask_input>

        - Make use of chat history to evaluate what is currently being asked and what has already been answered
        - Here is the conversation history that you need to take into account before answering:
        <history>
        {{HISTORY}}
        </history>

        - make sure to provide answer the sub-task with appropriate reasoning demanded in original_user_input
        """

        self.system_prompt = ""

        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'processPrompt',
                    'description': 'Analyze and process current sub-task input and provide structured output',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'type': 'string',
                                'description': 'The current sub-task input',
                            },
                            'output': {
                                'type': 'string',
                                'description': 'The detailed output of the current sub-task input',
                            },
                            'short_output': {
                                'type': 'string',
                                'description': 'The specific output the sub-task demands without details and other contexts.'
                            }
                        },
                        'required': ['input', 'output', 'short_output'],
                    },
                },
            }
        ]

    def is_streaming_enabled(self) -> bool:
        return self.streaming is True

    async def handle_request(
        self,
        original_user_input: str,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> Union[ConversationMessage, AsyncIterable[Any]]:
        try:
            self.original_user_input = original_user_input
            self.subtask_input = input_text
            self.set_history(chat_history)
            
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
                "stream": self.streaming,
                "tools": self.tools,
                "tool_choice":{"type": "function", "function": {"name": "processPrompt"}}
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

            tool_response = chat_completion.choices[0].message.tool_calls[0]

            if not tool_response or tool_response.function.name != "processPrompt":
                raise ValueError("Call to tool function processPrompt is missing")

            tool_input = json.loads(tool_response.function.arguments)

            assistant_message = tool_input['output']
            
            if not isinstance(assistant_message, str):
                raise ValueError('Unexpected response format from OpenAI API')

            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                original_user_input=self.original_user_input,
                short_output=tool_input['short_output'],
                tokens=len(assistant_message.split(' ')),
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
                if chunk.choices[0].delta.tool_calls:
                    tool_response = chunk.choices[0].delta.tool_calls[0]
                    accumulated_message.append(tool_response.function.arguments)

            # Store the complete message in the instance for later access if needed
            tool_input = json.loads(''.join(accumulated_message))
            if self.callbacks:
                self.callbacks.on_llm_new_token("\n")
                self.callbacks.on_llm_new_token(f"\nGenerated response from {self.name}")
                self.callbacks.on_llm_new_token("\n")
                self.callbacks.on_llm_new_token(tool_input['output'])
                tokens = len(tool_input['output'].split(' '))

            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                original_user_input=self.original_user_input,
                short_output=tool_input['short_output'],
                tokens=tokens,
                content=[{"text": ''.join(tool_input['output'])}]
            )

        except Exception as error:
            logger.error(f"Error getting stream from OpenAI model: {str(error)}")
            raise error