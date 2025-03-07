from typing import Dict, Any, AsyncIterable, Optional, Union, List
from dataclasses import dataclass, fields, asdict, replace
from loguru import logger
from orchestrator_types import ConversationMessage, ConversationRole, OrchestratorConfig, FinalResponse
from classifiers import Classifier,ClassifierResult
from agents import Agent, AgentResponse
from chat_storage import ChatStorage
from chat_storage import MemoryStorage

@dataclass
class Orchestrator:
    def __init__(self,
                 options: Optional[OrchestratorConfig] = None,
                 storage: Optional[ChatStorage] = None,
                 classifier: Optional[Classifier] = None):

        DEFAULT_CONFIG=OrchestratorConfig()

        if options is None:
            options = {}
        if isinstance(options, dict):
            # Filter out keys that are not part of OrchestratorConfig fields
            valid_keys = {f.name for f in fields(OrchestratorConfig)}
            options = {k: v for k, v in options.items() if k in valid_keys}
            options = OrchestratorConfig(**options)
        elif not isinstance(options, OrchestratorConfig):
            raise ValueError("options must be a dictionary or an OrchestratorConfig instance")

        self.config = replace(DEFAULT_CONFIG, **asdict(options))
        self.storage = storage

        self.agents: Dict[str, Agent] = {}
        self.storage = storage or MemoryStorage()

        if classifier:
            self.classifier = classifier
        else:
            raise ValueError("No classifier provided. Please provide a classifier.")
   
        self.execution_times: Dict[str, float] = {}


    def add_agent(self, agent: Agent):
        if agent.id in self.agents:
            raise ValueError(f"An agent with ID '{agent.id}' already exists.")
        self.agents[agent.id] = agent
        self.classifier.set_agents(self.agents)

    def get_all_agents(self) -> Dict[str, Dict[str, str]]:
        return {key: {
            "name": agent.name,
            "description": agent.description
        } for key, agent in self.agents.items()}

    async def dispatch_request_to_agent(self,
                                params: Dict[str, Any]) -> Union[
                                    ConversationMessage, AsyncIterable[Any]
                                ]:
        original_user_input = params['original_user_input']
        user_input = params['user_input']
        user_id = params['user_id']
        session_id = params['session_id']
        classifier_result:ClassifierResult = params['classifier_result']
        additional_params = params.get('additional_params', {})

        if not classifier_result.agent_selected:
            return "I'm sorry, but I need more information to understand your request. \
                Could you please be more specific?"

        agent_selected = classifier_result.agent_selected
        agent_chat_history = await self.storage.fetch_chat(user_id, session_id, agent_selected.id)

        response = await agent_selected.handle_request(
                                                   original_user_input,
                                                   user_input,
                                                   user_id,
                                                   session_id,
                                                   agent_chat_history,
                                                   additional_params)

        return response

    async def classify_request(self,
                             user_input: str,
                             user_id: str,
                             session_id: str) -> ClassifierResult:
        """Classify user request with conversation history."""
        try:
            chat_history = await self.storage.fetch_all_chats(user_id, session_id) or []
            classifier_result = await self.classifier.classify(user_input, chat_history)

            return classifier_result

        except Exception as error:
            logger.error(f"Error during intent classification: {str(error)}")
            raise error
        
    async def agent_handle_request(self,
                               original_user_input: str,
                               current_user_input: str,
                               user_id: str,
                               session_id: str,
                               classifier_result: ClassifierResult,
                               additional_params: Dict[str, str] = {}) -> AgentResponse:
        """Process agent response and handle chat storage."""
        try:
            agent_response = await self.dispatch_request_to_agent({
                "original_user_input": original_user_input,
                "user_input": current_user_input,
                "user_id": user_id,
                "session_id": session_id,
                "classifier_result": classifier_result,
                "additional_params": additional_params
            })

            logger.info(f"Output of the request is: {agent_response.content}")

            await self.save_message(
                ConversationMessage(
                    role=ConversationRole.USER.value,
                    original_user_input=original_user_input,
                    short_output="",
                    tokens=len(current_user_input.split(' ')),
                    content=[{'text': current_user_input}]
                ),
                user_id,
                session_id,
                classifier_result.agent_selected
            )
            if isinstance(agent_response, ConversationMessage):
                await self.save_message(agent_response,
                                    user_id,
                                    session_id,
                                    classifier_result.agent_selected)

            return AgentResponse(
                output=agent_response,
                streaming=classifier_result.agent_selected.is_streaming_enabled()
            )

        except Exception as error:
            logger.error(f"Error during agent processing: {str(error)}")
            raise error
        
    async def route_request(self,
                       user_input: str,
                       user_id: str,
                       session_id: str, 
                       request_id: str,
                       additional_params: Dict[str, str] = {}) -> AgentResponse:
        """Route user request to appropriate agent."""
        self.execution_times.clear()

        try:
            logger.info(f"User input: {user_input}")
            self.classifier.set_original_user_input(user_input)
            last_output_from_agent = ""
            final_response: List[FinalResponse] = []
            #---------------------Main Core Logic of Orchestrator Routing-------------------------
            while True:
                classifier_result = await self.classify_request(user_input, user_id, session_id)
                if not classifier_result.agent_selected:
                    return AgentResponse(
                        output=ConversationMessage(
                            original_user_input=self.classifier.original_user_input,
                            role=ConversationRole.ASSISTANT.value,
                            short_output="",
                            tokens=0,
                            content=[{'text': self.config.AGENT_NOT_SELECTED_LOG}]
                        ),
                        streaming=False
                    )

                logger.info(f"Agent Selected: {classifier_result.agent_selected.name} and Accuracy Score: {classifier_result.accuracy}")

                user_input = classifier_result.input + "\n" + last_output_from_agent
                logger.info(f"Current Input to be executed: {user_input}")

                agent_output = await self.agent_handle_request(
                        self.classifier.original_user_input,
                        user_input, 
                        user_id,
                        session_id, 
                        classifier_result,
                        additional_params
                )

                if isinstance(agent_output.output, ConversationMessage):
                        current_output = agent_output.output.content[0]['text']
                        tokens=agent_output.output.tokens
                else:
                        current_output = agent_output.output

                final_response.append(FinalResponse(
                    AGENT_OUTPUT=current_output,
                    AGENT_SELECTED=classifier_result.agent_selected.name,
                    OUTPUT_TOKENS=tokens,
                    REQUEST_ID=request_id
                ))

                if classifier_result.next_action == "respond_to_user":                
                    logger.info(f"Reponse from Agent: {current_output}")
                    break             
                elif classifier_result.next_action == "unknown":
                    logger.info(f"Reponse from Agent: {current_output}")
                    logger.info(f"No next tasks detected")
                    break
                else:
                    last_output_from_agent = agent_output.output.short_output
                    if classifier_result.next_action_input == "unkown":
                        user_input = current_output
                    else:
                        user_input = classifier_result.next_action_input
                    self.classifier.set_sutask_input(user_input)
                    logger.info(f"Performing next action with agent: {classifier_result.next_action}")
                    logger.info(f"Providing input to the agent: {user_input}")
            
            return final_response

        except Exception as error:
            return AgentResponse(
                output=self.config.ROUTING_ERROR_LOG or str(error),
                streaming=False
            )

    async def save_message(self,
                           message: ConversationMessage,
                           user_id: str, session_id: str,
                           agent: Agent):
        if agent and agent.save_chat:
            return await self.storage.save_message(user_id,
                                                        session_id,
                                                        agent.id,
                                                        message,
                                                        self.config.MAX_MESSAGE_PAIRS_PER_AGENT)