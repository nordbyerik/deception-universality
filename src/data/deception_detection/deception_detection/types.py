from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, Union

Judgement = Literal["honest", "deceptive", "ambiguous", "skip"] | str | int

Role = Literal["system", "user", "assistant"]

MODEL_TOKENS = {"llama": {}}


@dataclass
class Message:
    """
    We allow for messages from the same role to be adjacent.
    In this case, such messages are concatenated.
    """

    role: Role
    content: str
    detect: bool


Dialogue = list[Message]

TOKENS = {
    "llama2": {
        "system_start": "<s>[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",  # Continues in INST block or new [INST]
        "user_end": " [/INST]",
        "assistant_start": "",  # Immediately after [/INST]
        "assistant_end": " </s>",
        "bos": "<s>",
        "eos": "</s>",
    },
    "llama3": {
        "bos": "<|begin_of_text|>",
        "eos": "<|end_of_text|>",
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
        "eot": "<|eot_id|>",  # End of turn
    },
    "mistral": {
        "bos": "<s>",
        "eos": "</s>",
        "user_start": "[INST] ",
        "user_end": " [/INST]",
        "assistant_start": "",
        "assistant_end": "</s>",
    },
    "mixtral": {  # Same as Mistral
        "bos": "<s>",
        "eos": "</s>",
        "user_start": "[INST] ",
        "user_end": " [/INST]",
        "assistant_start": "",
        "assistant_end": "</s>",
    },
    "vicuna": {
        "system": "SYSTEM: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "bos": "<s>",
        "eos": "</s>",
    },
    "alpaca": {
        "instruction_start": "### Instruction:\n",
        "input_start": "### Input:\n",
        "response_start": "### Response:\n",
        "bos": "<s>",
        "eos": "</s>",
    },
    "phi2": {
        "user_start": "Instruct: ",
        "user_end": "\n",
        "assistant_start": "Output: ",
        "assistant_end": "\n",
    },
    "phi3": {
        "system_start": "<|system|>\n",
        "system_end": "<|end|>\n",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "<|end|>\n",
    },
    "gemma": {
        "bos": "<bos>",
        "eos": "<eos>",
        "user_start": "<start_of_turn>user\n",
        "user_end": "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "assistant_end": "<end_of_turn>\n",
    },
    "zephyr": {  # HuggingFace's chat format
        "system_start": "<|system|>\n",
        "system_end": "</s>\n",
        "user_start": "<|user|>\n",
        "user_end": "</s>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "</s>\n",
    },
    "chatml": {  # Used by many models including some OpenAI
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
    "command_r": {  # Cohere's Command R
        "bos": "<BOS_TOKEN>",
        "eos": "<EOS_TOKEN>",
        "system_start": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        "system_end": "<|END_OF_TURN_TOKEN|>",
        "user_start": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "user_end": "<|END_OF_TURN_TOKEN|>",
        "assistant_start": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "assistant_end": "<|END_OF_TURN_TOKEN|>",
    },
    "deepseek": {
        "bos": "<｜begin▁of▁sentence｜>",
        "user_start": "User: ",
        "user_end": "\n\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "<｜end▁of▁sentence｜>",
    },
    "qwen": {
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
    "yi": {
        "bos": "<|startoftext|>",
        "eos": "<|endoftext|>",
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
}


class GradingPromptGetter(Protocol):
    def __call__(
        self,
        input_dialogue: Dialogue,
        response: Message,
        metadata: dict[str, Any] | None,
    ) -> dict[str, str]: ...

    """
    Get a prompt for grading a given dialogue and response.
    Generally defined by `DialogueDataset.get_grading_prompt`.
    """


class Label(Enum):
    HONEST = "honest"
    DECEPTIVE = "deceptive"
    AMBIGUOUS = "ambiguous"
    SKIP = "skip"


# type alias for the output of `_get_dialogues()`
# dialogues, labels, metadata
# todo: convert to named tuple?
DialogueDataType = tuple[list[Dialogue], list[Label], dict[str, list[Any]] | None]
