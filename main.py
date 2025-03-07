from io import TextIOWrapper
from agents import TextClassifierAgent, TextClassifierAgentOptions, ReasoningAgent, ReasoningAgentOptions, DataRetrievalAgent, DataRetrievalAgentOptions
from orchestrator import Orchestrator
from orchestrator_types import FinalResponse, ExpectedResult
from chat_storage import MemoryStorage
from classifiers import OpenAIClassifier, OpenAIClassifierOptions
import asyncio
from typing import List
from loguru import logger
from guardrails.hub import DetectPII, DetectJailbreak, ProfanityFree
from guardrails import Guard

from dotenv import load_dotenv
import os
import random

# Load environment variables from .env file
load_dotenv()


class MainClass():
    def __init__(self):
        # Create classifier
        open_ai_classifier = OpenAIClassifier(
            OpenAIClassifierOptions(
                model=os.getenv('CLASSIFIER_MODEL'),
                api_key=os.getenv('CLASSIFIER_API_KEY')
            )
        )

        # Create Text Classification Agent
        text_classification_agent = TextClassifierAgent(TextClassifierAgentOptions(
                name="Text Classification Agent",
                description="Classifies given sentenence into provided classification label. \n Just provide answer either one of the classification label, do not provide any reasons. \n You perform only classification tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 5000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('TEXT_CLASSIFIER_MODEL'),
                api_key=os.getenv('TEXT_CLASSIFIER_API_KEY'),
                streaming=False
            )
        )

        # Create Reasoning Agent
        reasoning_agent = ReasoningAgent(ReasoningAgentOptions(
                name="Reasoning Agent",
                description="Evaluates given task and provides in depth reasons to justify arrived solution.\n You perform only reasoning tasks. \n You do not perform data retrieval.",
                save_chat=True,
                agent_config={
                    'maxTokens': 5000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('REASONING_MODEL'),
                api_key=os.getenv('REASONING_API_KEY'),
                streaming=False
            )
        )

        # Create Data Retrieval Agent
        data_retrieval_agent = DataRetrievalAgent(DataRetrievalAgentOptions(
                name="Data Retrieval Agent",
                description="Answer given question by providing in depth reasoning and knowledge using provided knowledge base or search tool. \n Include yes or no in your response if question asks for. \n You perform only data retrieval tasks. \n You do not perform reasoning tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 5000,
                    'temperature': None,
                    'topP': None,
                    'stopSequences': None
                },
                model=os.getenv('DATA_RETRIEVER_MODEL'),
                api_key=os.getenv('DATA_RETRIEVER_API_KEY'),
                streaming=False,
                use_google_tool=True
            )
        )

        # Create AgentOrchestrator
        self.agent_orchestrator = Orchestrator(storage=MemoryStorage(),classifier=open_ai_classifier)

        # Add agents
        self.agent_orchestrator.add_agent(text_classification_agent)
        self.agent_orchestrator.add_agent(reasoning_agent)
        self.agent_orchestrator.add_agent(data_retrieval_agent)

    def run(self, user_input: str, user_id: str, session_id: str, request_id: str):
        return asyncio.run(self.agent_orchestrator.route_request(user_input,user_id,session_id,request_id))
    
    def evaluationMetric(self, user_input: str, results: List[FinalResponse], expectedResult: ExpectedResult, file: TextIOWrapper):
        try:
            total_output_token = 0
            agents_called_set = set()
            agent_output = []
            for result in results:
                agents_called_set.add(result.AGENT_SELECTED)
                agent_output.append(result.AGENT_OUTPUT)
                total_output_token = total_output_token + result.OUTPUT_TOKENS


            agent_result_set = set()
            for resultKeyword in expectedResult.RESULTS:
                for result in results:
                    if resultKeyword.lower() in result.AGENT_OUTPUT.lower():
                        agent_result_set.add(resultKeyword)


            print(f"\n----------------------------- Start of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------")
            print(f"User input: {user_input}")
            print(f"Agent response: {agent_output}")
            print(f"Number of agent calls: {len(results)}")
            print(f"Agents called: {list(agents_called_set)}")
            print(f"Total output tokens: {total_output_token}")
            print(f"----------------------------- End of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------\n")

            file.write(
                f'''
                        ----------------------------- Start of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------
                        User input: {user_input}
                        Agent response: {agent_output}
                        Number of agent calls: {len(results)}
                        Agents called: {list(agents_called_set)}
                        Total output tokens: {total_output_token}
                        ----------------------------- End of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------\n
                '''
            )
            file.flush()
        except:
            logger.error("Unexpeted error from agent")


if __name__ == '__main__':
    try:
        file = open("log.txt", "w")

        mainClass = MainClass()
        
        user_id="test-user-1"
        session_id="123456"

        request_prefix = user_id + "-" + session_id
        i = 1
        
        input_guard = Guard().use_many(
            DetectJailbreak(on_fail="exception"),
            DetectPII(["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="exception")
        )

        output_guard = Guard().use_many(
            ProfanityFree(on_fail="exception"),
        )


        while True:

            print(f"Performing request: {i}")
            user_input = input("Enter your query: ")
            if user_input == "exit":
                break
            else:
                try:
                    # expected_calls = int(input("Number of agent calls expected: "))
                    input_guard.validate(user_input)
                    results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(i))
                    for result in results:
                        output_guard.validate(result.AGENT_OUTPUT)

                    expectedResult = ExpectedResult(
                        NUMBER_OF_AGENT_CALL=1,
                        AGENTS=[],
                        RESULTS=[]
                    )
                    mainClass.evaluationMetric(user_input, results, expectedResult, file)
                except Exception as error:
                    logger.info(f"Error: {error}")    
            i = i + 1

        file.close()
    except Exception as error:
        logger.error(f"Error in main method: {error}")

    



