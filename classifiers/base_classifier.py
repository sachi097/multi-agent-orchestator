from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from orchestrator_types import ConversationMessage
from agents import Agent


@dataclass
class ClassifierResult:
    agent_selected: Optional[Agent]
    accuracy: float
    action: str 
    next_action: str
    next_action_input: str

class Classifier(ABC):
    def __init__(self):
        self.agent_descriptions = ""
        self.chat_history = ""
        response_format = {"agent_selected":"", "accuracy":"", "input":"", "action":"", "next_action":"", "next_action_input": ""}
        self.prompt_template = """
                                    You are AgentMatcher, an intelligent assistant designed to analyze user queries and match them with
                                    the most suitable agent or department. Your task is to understand the user's request,
                                    identify key entities and intents, and determine which agent or department would be best equipped
                                    to handle the query. You should divided the query into sub-tasks if needed and chain these tasks with suitable agents to achieve final answer.

                                    Important: The user's input may be a follow-up response to a previous interaction.
                                    The conversation history, including the name of the previously selected agent, is provided.
                                    If the user's input appears to be a continuation of the previous conversation
                                    (e.g., "yes", "ok", "I want to know more", "1"), select the same agent as before.

                                    Analyze the user's input and categorize it into one of the following agent types:
                                    <agents>
                                    {{AGENT_DESCRIPTIONS}}
                                    </agents>
                                    If you are unable to select an agent put "unknown"

                                    Guidelines for classification:

                                        Agent Type: Choose the most appropriate agent type based on the nature of the query.
                                        For follow-up responses, use the same agent type as the previous interaction.
                                        Priority: Assign based on urgency and impact.
                                            High: Issues affecting service, billing problems, or urgent technical issues
                                            Medium: Non-urgent product inquiries, sales questions
                                            Low: General information requests, feedback
                                        Key Entities: Extract important nouns, product names, or specific issues mentioned.
                                        For follow-up responses, include relevant entities from the previous interaction if applicable.
                                        For follow-ups, relate the intent to the ongoing conversation.
                                        accuracy: Indicate how confident you are in the classification.
                                            High: Clear, straightforward requests or clear follow-ups
                                            Medium: Requests with some ambiguity but likely classification
                                            Low: Vague or multi-faceted requests that could fit multiple categories
                                        Is Followup: Indicate whether the input is a follow-up to a previous interaction.

                                    Handle variations in user input, including different phrasings, synonyms,
                                    and potential spelling errors.
                                    For short responses like "yes", "ok", "I want to know more", or numerical answers,
                                    treat them as follow-ups and maintain the previous agent selection.

                                    Here is the conversation history that you need to take into account before answering:
                                    <history>
                                    {{HISTORY}}
                                    </history>

                                    Examples:

                                    1. Initial query with no context:
                                    User: "What are the symptoms of the flu?"

                                    userinput: What are the symptoms of the flu?
                                    agent_selected: agent-name
                                    accuracy: 0.95

                                    2. Context switching example between a TechAgent and a BillingAgent:
                                    Previous conversation:
                                    User: "How do I set up a wireless printer?"
                                    Assistant: [agent-a]: To set up a wireless printer, follow these steps:
                                    1. Ensure your printer is Wi-Fi capable.
                                    2. Connect the printer to your Wi-Fi network.
                                    3. Install the printer software on your computer.
                                    4. Add the printer to your computer's list of available printers.
                                    Do you need more detailed instructions for any of these steps?
                                    User: "Actually, I need to know about my account balance"

                                    userinput: Actually, I need to know about my account balance</userinput>
                                    agent_selected: agent-name
                                    accuracy: 0.9

                                    3. Follow-up query example for the same agent:
                                    Previous conversation:
                                    User: "What's the best way to lose weight?"
                                    Assistant: [agent-name-1]: The best way to lose weight typically involves a combination
                                    of a balanced diet and regular exercise.
                                    It's important to create a calorie deficit while ensuring you're getting proper nutrition.
                                    Would you like some specific tips on diet or exercise?
                                    User: "Yes, please give me some diet tips"

                                    userinput: Yes, please give me some diet tips
                                    agent_selected: agent-name-1
                                    accuracy: 0.95

                                    4. Multiple context switches with final follow-up:
                                    Conversation history:
                                    User: "How much does your premium plan cost?"
                                    Assistant: [agent-name-a]: Our premium plan is priced at $49.99 per month.
                                    This includes features such as unlimited storage, priority customer support,
                                    and access to exclusive content. Would you like me to go over the benefits in more detail?
                                    User: "No thanks. Can you tell me about your refund policy?"
                                    Assistant: [agent-name-b]: Certainly! Our refund policy allows for a full refund within 30 days
                                    of purchase if you're not satisfied with our service. After 30 days, refunds are prorated based
                                    on the remaining time in your billing cycle. Is there a specific concern you have about our service?
                                    User: "I'm having trouble accessing my account"
                                    Assistant: [agent-name-c]: I'm sorry to hear you're having trouble accessing your account.
                                    Let's try to resolve this issue. Can you tell me what specific error message or problem
                                    you're encountering when trying to log in?
                                    User: "It says my password is incorrect, but I'm sure it's right"

                                    userinput: It says my password is incorrect, but I'm sure it's right
                                    agent_selected: agent-name-c
                                    accuracy: 0.9

                                    Skip any preamble and provide only the response in the specified format.

                                    ###Guidelines###
                                    - Sometimes you might have to use multiple agent's to solve user's input. You have to do that in a loop.
                                    - The original userinput could have multiple tasks, you will use the context to understand the previous actions taken and the next steps you should take.
                                    - Read the context, take your time to understand, see if there were many tasks and if you executed them all
                                    - If there are no actions to be taken, then make the action "respond_to_user" with your final thoughts combining all previous responses as input.
                                    - If there are no actions to be taken, then put "unknown" in "next_action_input"
                                    - Respond with "respond_to_user" only when there are no agents to select from or there is no next_action
                                    - Respond with "unknown" in "next_action_input" only when there are no agents to select from or there is no next_action
                                    - If there are actions to be taken next, then analyze the next action and categorize it into one of the following agent types:
                                        <agents>
                                        {{AGENT_DESCRIPTIONS}}
                                        </agents>
                                    - If there are actions to be taken next, then for the next agent selected provide the input in "next_action_input"
                                    If you are unable to select an agent put "unknown"
                                    - You will return the agent name in the form of {response_format}
                                    - Always return valid JSON like {response_format} and nothing else. 
                                    """
        self.system_prompt = ""
        self.agents: Dict[str, Agent] = {}

    def set_agents(self, agents: Dict[str, Agent]) -> None:
        self.agent_descriptions = "\n\n".join(f"{agent.id}:{agent.description}"
                                              for agent in agents.values())
        self.agents = agents

    def set_history(self, messages: List[ConversationMessage]) -> None:
        self.chat_history = self.format_messages(messages)

    @staticmethod
    def format_messages(messages: List[ConversationMessage]) -> str:
        return "\n".join([
            f"{message.role}: {' '.join([message.content[0]['text']])}" for message in messages
        ])

    async def classify(self,
                       input_text: str,
                       chat_history: List[ConversationMessage]) -> ClassifierResult:
        self.set_history(chat_history)
        self.update_system_prompt()
        return await self.make_request(input_text)

    @abstractmethod
    async def make_request(self,input_text: str) -> ClassifierResult:
        pass

    def update_system_prompt(self) -> None:
        all_variables: Dict[str, Union[str, List[str]]] = {
            "AGENT_DESCRIPTIONS": self.agent_descriptions,
            "HISTORY": self.chat_history,
        }
        self.system_prompt = self.replace_placeholders(self.prompt_template, all_variables)

    @staticmethod
    def replace_placeholders(template: str, variables: Dict[str, Union[str, List[str]]]) -> str:

        return re.sub(r'{{(\w+)}}',
                      lambda m: '\n'.join(variables.get(m.group(1), [m.group(0)]))
                      if isinstance(variables.get(m.group(1)), list)
                      else variables.get(m.group(1), m.group(0)), template)

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        if not agent_id:
            return None
        my_agent_id = agent_id.split(" ")[0].lower()
        return self.agents.get(my_agent_id)