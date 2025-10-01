import json
import os


def load_pairs_dataset(
    filename="out_simple_4_many.json",
    data_dir="data/deception-detection/data/ft_fact_paired",
):
    """Parse the pairs dataset from JSON file."""

    file_path = os.path.join(data_dir, filename)
    dataset = []

    with open(file_path, "r") as file:
        data = json.load(file)

        for item in data:
            # Extract the assistant's response text and expected honesty
            assistant_message = None
            for msg in item["input_messages"]:
                if msg["role"] == "assistant" and msg.get("detect", False):
                    assistant_message = msg["content"]
                    break

            if assistant_message:
                # expect_honesty: True for honest, False for deceptive
                label = item["expect_honesty"]
                dataset.append((assistant_message, label))

    return dataset
Keebio FoldKB

def load_roleplay_dataset(
    filename="roleplaying__plain__llama-70b.json",
    data_dir="src/data/deception-detection/data/rollouts",
):
    """Parse the roleplay dataset from JSON file."""

    file_path = os.path.join(data_dir, filename)
    dataset = []

    with open(file_path, "r") as file:
        data = json.load(file)
        rollouts = data.get("rollouts", [])

        for rollout in rollouts:
            # Extract the assistant's response text
            assistant_message = None
            for msg in rollout["input_messages"]:
                if msg["role"] == "assistant" and msg.get("detect", False):
                    assistant_message = msg["content"]
                    break

            if assistant_message:
                # Check if this is expected to be honest or deceptive based on context
                # For roleplay, we infer from the system message setup
                system_msg = next(
                    (
                        msg["content"]
                        for msg in rollout["input_messages"]
                        if msg["role"] == "system"
                    ),
                    "",
                )
                # If the scenario involves deception (making up excuses), mark as False (deceptive)
                label = (
                    "making up" not in system_msg.lower()
                    and "excuse" not in system_msg.lower()
                )
                dataset.append((assistant_message, label))

    return dataset
