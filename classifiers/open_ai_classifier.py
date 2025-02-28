import json
from typing import Optional, Dict, Any
from openai import OpenAI
from loguru import logger
from classifiers import Classifier, ClassifierResult

OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini"

class OpenAIClassifierOptions:
    def __init__(self,
                 api_key: str,
                 model: Optional[str] = None,
                 classifier_config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.model = model
        self.classifier_config = classifier_config or {}

class OpenAIClassifier(Classifier):
    def __init__(self, options: OpenAIClassifierOptions):
        super().__init__()

        if not options.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=options.api_key)
        self.model = options.model or OPENAI_MODEL_ID_GPT_O_MINI

        default_max_tokens = 1000
        self.classifier_config = {
            'max_tokens': options.classifier_config.get('max_tokens', default_max_tokens),
            'temperature': options.classifier_config.get('temperature', 0.0),
            'top_p': options.classifier_config.get('top_p', 0.9),
            'stop': options.classifier_config.get('stop_sequences', []),
        }

        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'processPrompt',
                    'description': 'Analyze and process user input and provide structured output',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'type': 'string',
                                'description': 'The original input of the user',
                            },
                            'agent_selected': {
                                'type': 'string',
                                'description': 'The name of the agent selected',
                            },
                            'accuracy': {
                                'type': 'number',
                                'description': 'Accuracy score between 0 and 1',
                            },
                            'action': {
                                'type': 'string',
                                'description': 'The action you are taking for the user input'
                            }, 
                            "next_action": {
                                'type': 'string',
                                'description': 'The next action that needs to be taken'
                            },
                            "next_action_input": {
                                'type': 'string',
                                'description': 'The input to next action agent'
                            }
                        },
                        'required': ['input', 'agent_selected', 'accuracy', 'action', 'next_action', 'next_action_input'],
                    },
                },
            }
        ]

    async def make_request(self,input_text: str) -> ClassifierResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.classifier_config['max_tokens'],
                temperature=self.classifier_config['temperature'],
                top_p=self.classifier_config['top_p'],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "processPrompt"}}
            )

            tool_response = response.choices[0].message.tool_calls[0]

            if not tool_response or tool_response.function.name != "processPrompt":
                raise ValueError("Call to tool function processPrompt is missing")

            tool_input = json.loads(tool_response.function.arguments)

            if tool_input['next_action_input'] == '':
                tool_input['next_action_input'] = "unknown"

            intent_classifier_result = ClassifierResult(
                agent_selected=self.get_agent_by_id(tool_input['agent_selected']),
                accuracy=float(tool_input['accuracy']),
                action=tool_input['action'],
                next_action=tool_input['next_action'],
                next_action_input=tool_input['next_action_input']
            )

            return intent_classifier_result

        except Exception as error:
            logger.error(f"Request processing error: {str(error)}")
            raise error