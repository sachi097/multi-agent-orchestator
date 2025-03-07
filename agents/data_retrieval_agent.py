from typing import Dict, List, Union, AsyncIterable, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from phi.tools.googlesearch import GoogleSearch
from phi.agent import Agent as phiAgent
from phi.agent import RunResponse
from phi.model.openai import OpenAIChat
from agents import Agent, AgentOptions
from orchestrator_types import (
    ConversationMessage,
    ConversationRole
)
from loguru import logger
from retrievers import Retriever, JSONRetriever

OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"

@dataclass
class DataRetrievalAgentOptions(AgentOptions):
    api_key: str = None
    model: Optional[str] = None
    streaming: Optional[bool] = None
    agent_config: Optional[Dict[str, Any]] = None
    retriever: Optional[Retriever] = None
    client: Optional[Any] = None
    use_google_tool: bool = True


class OutputFormat(BaseModel):
    input: str = Field(..., description="The current sub-task input.")
    output: str = Field(..., description="The detailed output of the current sub-task input")
    short_output: str = Field(..., description="The specific output the current sub-task demands without details and other contexts. If sub-task asks for records or list or texts or sentences or similar thing then include these in this field. For Example: sub-task: 'fetch top 5 sentences from google search and classify them as hate-speech or not-speech', then 'short_ouput': 'the top 5 sentences retrieved without classification of the sentences to hate-speech or not-hate-speech'")

class DataRetrievalAgent(Agent):
    def __init__(self, options: DataRetrievalAgentOptions):
        super().__init__(options)
        if not options.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = options.model or OPENAI_MODEL_ID_GPT_O_MINI
        self.api_key = options.api_key
        self.streaming = options.streaming or False
        self.callbacks = options.callbacks

        # Initialize system prompt
        self.prompt_template = f"""You are a {self.name}.
        {self.description} 
        You specialize only in tasks to Provide helpful and accurate information based on your expertise and make use of the search engine to fetch the relevant data.
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

        - make sure to provide answer the sub-task with appropriate retrievals labels in original_user_input
        """

        if options.client:
            self.client = options.client
        else:
            if options.use_google_tool:
                self.client = phiAgent(
                        model=OpenAIChat(id="gpt-4o-mini", api_key=self.api_key),
                        show_tool_calls=False,
                        markdown=True,
                        description=self.description,
                        instructions=[self.prompt_template],
                        tools=[GoogleSearch()],
                        response_model=OutputFormat,
                        structured_outputs=True
                    )
            else:
                self.retriever: Optional[Retriever] = JSONRetriever().get_retriever(options.api_key)
                self.client = phiAgent(
                        model=OpenAIChat(id="gpt-4o-mini", api_key=self.api_key),
                        show_tool_calls=False,
                        markdown=True,
                        description=self.description,
                        instructions=[self.prompt_template],
                        # Add the knowledge base to the agent
                        knowledge=self.retriever,
                        response_model=OutputFormat,
                        structured_outputs=True
                    )

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

        self.system_prompt = ""

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
            chat_completion: RunResponse = self.client.run(**request_options)
            assistant_message: OutputFormat = chat_completion.content
            
            if not isinstance(assistant_message, OutputFormat):
                raise ValueError('Unexpected response format from OpenAI API')
            
            tokens = 0
            if len(chat_completion.metrics['output_tokens']) >= 2:
                tokens = chat_completion.metrics['output_tokens'][1]
            else:
                tokens = len(assistant_message.output.split(' '))

            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                original_user_input=self.original_user_input,
                short_output=assistant_message.short_output,
                tokens=tokens,
                content=[{"text": assistant_message.output}]
            )

        except Exception as error:
            logger.error(f'Error in OpenAI API call: {str(error)}')
            raise error

    async def handle_streaming_response(self, request_options: Dict[str, Any]) -> ConversationMessage:
        try:
            request_options['stream'] = False
            streams: RunResponse = self.client.run(**request_options)
            assistant_message: OutputFormat = streams.content
            accumulated_message = [assistant_message.output]

            if self.callbacks:
                self.callbacks.on_llm_new_token("\n")
                self.callbacks.on_llm_new_token(f"\nGenerated response from {self.name}")
                self.callbacks.on_llm_new_token("\n")
                self.callbacks.on_llm_new_token(assistant_message.output)
            
            tokens = 0
            if len(streams.metrics['output_tokens']) >= 2:
                tokens = streams.metrics['output_tokens'][1]
            else:
                tokens = len(assistant_message.output.split(' '))

            # Store the complete message in the instance for later access if needed
            return ConversationMessage(
                role=ConversationRole.ASSISTANT.value,
                original_user_input=self.original_user_input,
                short_output=assistant_message.short_output,
                tokens=tokens,
                content=[{"text": ''.join(accumulated_message)}]
            )

        except Exception as error:
            logger.error(f"Error getting stream from OpenAI model: {str(error)}")
            raise error