from io import TextIOWrapper
from agents import TextClassifierAgent, TextClassifierAgentOptions, ReasoningAgent, ReasoningAgentOptions, DataRetrievalAgent, DataRetrievalAgentOptions
from orchestrator import Orchestrator
from orchestrator_types import FinalResponse, ExpectedResult
from chat_storage import MemoryStorage
from classifiers import OpenAIClassifier, OpenAIClassifierOptions
import asyncio
from typing import List
from loguru import logger
import dspy
from dspy.datasets import HotPotQA
from dspy.datasets.gsm8k import GSM8K

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
                streaming=False
            )
        )

        # Create Reasoning Agent
        reasoning_agent = ReasoningAgent(ReasoningAgentOptions(
                name="Reasoning Agent",
                description="Evaluates given task and provides in depth reasons to justify arrived solution.  You perform only reasoning tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 1000,
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
                description="Answer given question by providing in depth reasoning and knowledge using provided knowledge base or search tool. Include yes or no in your response if question asks for. You perform only data retrieval tasks.",
                save_chat=True,
                agent_config={
                    'maxTokens': 1000,
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
            print(f"Number of expected agent calls: {expectedResult.NUMBER_OF_AGENT_CALL}")
            print(f"Number of actual agent calls: {len(results)}")
            print(f"Efficiency of agent calls: {(len(results) / expectedResult.NUMBER_OF_AGENT_CALL) * 100} %")
            print(f"Agents expected to be called: {expectedResult.AGENTS}")
            print(f"Agents actually called: {list(agents_called_set)}")
            print(f"Expected result set: {expectedResult.RESULTS}")
            print(f"Actual result set: {list(agent_result_set)}")
            print(f"Total output tokens: {total_output_token}")
            print(f"----------------------------- End of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------\n")

            file.write(
                f'''
                        ----------------------------- Start of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------
                        User input: {user_input}
                        Agent response: {agent_output}
                        Number of expected agent calls: {expectedResult.NUMBER_OF_AGENT_CALL}
                        Number of actual agent calls: {len(results)}
                        Efficiency of agent calls: {(len(results) / expectedResult.NUMBER_OF_AGENT_CALL) * 100} %
                        Agents expected to be called: {expectedResult.AGENTS}
                        Agents actually called: {list(agents_called_set)}
                        Expected result set: {expectedResult.RESULTS}
                        Actual result set: {list(agent_result_set)}
                        Total output tokens: {total_output_token}
                        ----------------------------- End of Evaluation Metric for Request : {results[0].REQUEST_ID} --------------------------\n
                '''
            )
        except:
            logger.error("Unexpeted error from agent")


if __name__ == '__main__':
    try:
        file = open("log.txt", "w")
        
        # init GSM8k dataset
        datasetGSM = GSM8K()
        devGSMSet = [x.with_inputs('question') for x in datasetGSM.dev]
        
        # init HotPotQA dataset
        datasetHotQA = HotPotQA(dev_size=50, test_size=0)
        devHotQASet = [x.with_inputs('question') for x in datasetHotQA.dev]

        mainClass = MainClass()
        
        user_id="test-user-1"
        session_id="123456"

        request_prefix = user_id + "-" + session_id

        ## Easy tasks
        
        # 1. First test case
        user_input = "Classify below Sentence whether it is Positive or Negative. Answer Positive or Negative \nSentence: I am feeling so good today."
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=1,
            AGENTS=['Text Classification Agent'],
            RESULTS=['Positive']
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 2. Second test case from GSM8K - grade-school-math
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        user_input = "Evaluate A carnival snack booth made $50 selling popcorn each day. It made three times as much selling cotton candy. For a 5-day activity, the booth has to pay $30 rent and $75 for the cost of the ingredients. How much did the booth earn for 5 days after paying the rent and the cost of ingredients?"
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=1,
            AGENTS=['Reasoning Agent'],
            RESULTS=['895']
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 3. Random test case from GSM8K dataset
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_gsm_quetsion_index = random.randint(0, len(devGSMSet)-1)
        user_input = "Evaluate " + devGSMSet[random_gsm_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=1,
            AGENTS=['Reasoning Agent'],
            RESULTS=[devGSMSet[random_gsm_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 4. Known test case from HotPotQA
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        user_input = "What position on the Billboard Top 100 did Alison Moyet's late summer hit achieve?"
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=1,
            AGENTS=['Data Retrieval Agent'],
            RESULTS=[]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 5. Random test case from HotPotQA dataset
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_qa_quetsion_index = random.randint(0, len(devHotQASet)-1)
        user_input = devHotQASet[random_qa_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=1,
            AGENTS=['Data Retrieval Agent'],
            RESULTS=[devHotQASet[random_qa_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        ## Complex Tasks

        # 6. Test with classification and reasoning tasks
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_gsm_quetsion_index = random.randint(0, len(devGSMSet)-1)
        user_input = '''
                Is below sentence hate or not-hate speech? Answer Yes or No
                \nI personally think she sounds like a strangled cat.   
        ''' +  " And Evaluate " + devGSMSet[random_gsm_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=2,
            AGENTS=['Text Classification Agent', 'Reasoning Agent'],
            RESULTS=['Yes', devGSMSet[random_gsm_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 7. Test with classification and retrieval tasks
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_qa_quetsion_index = random.randint(0, len(devHotQASet)-1)
        user_input = '''
                Perform sentiment analysis on below sentence whether it is Happy or Sad. Answer Happy or Sad \nSentence: sad to say, she never lived to see it.   
        ''' +  " And Find " + devHotQASet[random_qa_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=2,
            AGENTS=['Text Classification Agent', 'Data Retrieval Agent'],
            RESULTS=['Sad', devHotQASet[random_qa_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 8. Test with classification, reasoning and retrieval tasks
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_gsm_quetsion_index = random.randint(0, len(devGSMSet)-1)
        random_qa_quetsion_index = random.randint(0, len(devHotQASet)-1)
        user_input = '''
                Perform spam analysis on below sentence whether it is Spam or Not-spam. Answer Yes or No \nSentence: You have an outstanding tax refund of $2,560. Follow these instructions to claim your refund at: https://gov.taxrefunds.irs.   
        ''' + " And Find " + devHotQASet[random_qa_quetsion_index]['question'] + " And The last task is to Evaluate " + devGSMSet[random_gsm_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=3,
            AGENTS=['Text Classification Agent', 'Data Retrieval Agent', 'Reasoning Agent'],
            RESULTS=['Yes', devHotQASet[random_qa_quetsion_index]['answer'], devGSMSet[random_gsm_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 9. Test with classification, reasoning and retrieval tasks
        session_id="123456" + str(random.randint(1, 100))
        request_prefix = user_id + "-" + session_id
        random_gsm_quetsion_index = random.randint(0, len(devGSMSet)-1)
        random_qa_quetsion_index = random.randint(0, len(devHotQASet)-1)
        user_input = '''
                Perform spam analysis on below sentence whether it is Spam or Not-spam. Answer Yes or No \nSentence: Hey sachin how are doing today its been long we spoke.   
        ''' + " And Find " + devHotQASet[random_qa_quetsion_index]['question'] + " And The last task is to Evaluate " + devGSMSet[random_gsm_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=3,
            AGENTS=['Text Classification Agent', 'Data Retrieval Agent', 'Reasoning Agent'],
            RESULTS=['No', devHotQASet[random_qa_quetsion_index]['answer'], devGSMSet[random_gsm_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        # 10. Test with classification, reasoning and retrieval tasks
        random_gsm_quetsion_index = random.randint(0, len(devGSMSet)-1)
        random_qa_quetsion_index = random.randint(0, len(devHotQASet)-1)
        user_input = '''
                Classify below sentence whether it is Positive or Negative. Answer Postive or Negative \nSentence: I am going to curse you very badly.   
        ''' + " And Find " + devHotQASet[random_qa_quetsion_index]['question'] + " And The last task is to Evaluate " + devGSMSet[random_gsm_quetsion_index]['question']
        results = mainClass.run(user_input, user_id, session_id, request_prefix.lower() + str(random.randint(1, 10)))
        expectedResult = ExpectedResult(
            NUMBER_OF_AGENT_CALL=3,
            AGENTS=['Text Classification Agent', 'Data Retrieval Agent', 'Reasoning Agent'],
            RESULTS=['Negative', devHotQASet[random_qa_quetsion_index]['answer'], devGSMSet[random_gsm_quetsion_index]['answer']]
        )
        mainClass.evaluationMetric(user_input, results, expectedResult, file)

        file.close()
    except Exception as error:
        logger.error(f"Error in main method: {error}")

    



