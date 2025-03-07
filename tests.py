from main import MainClass
from orchestrator_types import ExpectedResult
from loguru import logger
from dspy.datasets import HotPotQA
from dspy.datasets.gsm8k import GSM8K

from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()

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
                Classify below sentence whether it is Positive or Negative. Answer Positive or Negative \nSentence: I am going to curse you very badly.   
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