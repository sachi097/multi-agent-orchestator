from agents import TextClassifierAgent, TextClassifierAgentOptions, ReasoningAgent, ReasoningAgentOptions, DataRetrievalAgent, DataRetrievalAgentOptions, AgentCallbacks, AgentResponse
from orchestrator import Orchestrator
from orchestrator_types import ConversationMessage
from chat_storage import MemoryStorage
from classifiers import OpenAIClassifier, OpenAIClassifierOptions
import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from guardrails.hub import DetectPII, DetectJailbreak, ProfanityFree
from guardrails import Guard

from threading import Thread

from dotenv import load_dotenv
import os
import random

# Load environment variables from .env file
load_dotenv()

input_guard = Guard().use_many(
    DetectJailbreak(on_fail="exception"),
    DetectPII(["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="exception")
)

output_guard = Guard().use_many(
    ProfanityFree(on_fail="exception"),
)

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    user_input: str
    user_id: str
    session_id: str

class StreamHandler(AgentCallbacks):
    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue
        self._stop_signal = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._queue.put_nowait(token)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("generation started")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\n\ngeneration ended")

        self._queue.put_nowait(self._stop_signal)


class OrchestratorManager():

    def __init__(self) -> None:
        self.stream_queue = asyncio.Queue()
        self.agent_orchestrator: Orchestrator = None
        self.setup_orchestrator()

    def setup_orchestrator(self):
        # Create classifier
        open_ai_classifier = OpenAIClassifier(
            OpenAIClassifierOptions(
                model=os.getenv('CLASSIFIER_MODEL'),
                api_key=os.getenv('CLASSIFIER_API_KEY')
            )
        )

        my_handler = StreamHandler(self.stream_queue)

        # Create Text Classification Agent
        text_classification_agent = TextClassifierAgent(TextClassifierAgentOptions(
                name="Text Classification Agent",
                description="Classifies given sentenence into provided classification label. Just provide answer either one of the classification label, do not provide any reasons. You perform only classification tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 1000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('TEXT_CLASSIFIER_MODEL'),
                api_key=os.getenv('TEXT_CLASSIFIER_API_KEY'),
                streaming=True,
                callbacks=my_handler
            )
        )

        # Create Reasoning Agent
        reasoning_agent = ReasoningAgent(ReasoningAgentOptions(
                name="Reasoning Agent",
                description="Evaluates given task and provides in depth reasons to justify arrived solution. You perform only reasoning tasks. You do not perform data retrieval.",
                save_chat=True,
                agent_config={
                    'maxTokens': 1000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('REASONING_MODEL'),
                api_key=os.getenv('REASONING_API_KEY'),
                streaming=True,
                callbacks=my_handler
            )
        )

        # Create Data Retrieval Agent
        data_retrieval_agent = DataRetrievalAgent(DataRetrievalAgentOptions(
                name="Data Retrieval Agent",
                description="Answer given question by providing in depth reasoning and knowledge using provided knowledge base or search tool. Include yes or no in your response if question asks for. You perform only data retrieval tasks. You do not perform reasoning tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 1000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('DATA_RETRIEVER_MODEL'),
                api_key=os.getenv('DATA_RETRIEVER_API_KEY'),
                streaming=True,
                use_google_tool=True,
                callbacks=my_handler
            )
        )

        # Create AgentOrchestrator
        self.agent_orchestrator = Orchestrator(storage=MemoryStorage(),classifier=open_ai_classifier)
        self.agent_orchestrator.add_agent(text_classification_agent)
        self.agent_orchestrator.add_agent(reasoning_agent)
        self.agent_orchestrator.add_agent(data_retrieval_agent)


active_connections: Dict[str, OrchestratorManager] = {}

async def begin_generation(query, user_id, session_id, stream_queue, request_id, agent_orchestrator):
    try:
        orchestrator = agent_orchestrator.agent_orchestrator
        response = await orchestrator.route_request(query, user_id, session_id, request_id)
        if isinstance(response, AgentResponse) and response.streaming is False:
            if isinstance(response.output, str):
                output_guard.validate(response.output)
                stream_queue.put_nowait(response.output)
            elif isinstance(response.output, ConversationMessage):
                output_guard.validate(response.output.content[0].get('text'))
                stream_queue.put_nowait(response.output.content[0].get('text'))
    except Exception as e:
        print(f"Error in begin_generation: {e}")
        response = "Something went wrong"
        error = str(e).lower()
        if 'profanity' in error:
            response = "Unable to process request: Personally Identifiable Information Detected"
        stream_queue.put_nowait(str(response))
    finally:
        stream_queue.put_nowait(None)

def chat_generator(query, user_id, session_id, request_id, agent_orchestrator):
    stream_queue = agent_orchestrator.stream_queue
    Thread(target=lambda: asyncio.run(begin_generation(query, user_id, session_id, stream_queue, request_id, agent_orchestrator))).start()
    while True:
        try:
            try:
                value = stream_queue.get_nowait()
                if value is None:
                    break
                yield value
                stream_queue.task_done()
            except asyncio.QueueEmpty:
                pass
        except Exception as e:
            print(f"Error in chat_generator: {str(e)}")
            break

@app.post("/orchestrated_chat")
def orchestrated_chat(body: RequestBody):
    try:
        input_guard.validate(body.user_input)
        request_prefix = body.user_id + "-" + body.session_id
        request_id = request_prefix + str(random.randint(1, 10))
        agent_orchestrator = None
        if body.user_id in active_connections.keys():
            agent_orchestrator = active_connections[body.user_id]
        else:
            agent_orchestrator = OrchestratorManager()
            active_connections[body.user_id] = agent_orchestrator
        return StreamingResponse(chat_generator(body.user_input, body.user_id, body.session_id, request_id, agent_orchestrator), media_type="text/event-stream")
    except Exception as error:
        response = "Something went wrong"
        error = str(error).lower()
        if 'jailbreak' in error:
            response = "Unable to process request: Jailbreak Detected"
        elif 'pii' in error:
            response = "Unable to process request: Personally Identifiable Information Detected"
        return StreamingResponse(response, media_type="text/event-stream")
