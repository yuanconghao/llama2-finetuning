from datasets import load_dataset
import json
import os

dataset = load_dataset("jayantdocplix/medical_dataset_chat", "main")

dataset_splits = {"train": dataset["train"], "test": dataset["test"]}
print(dataset_splits)


def main():
    if not os.path.exists("data_med"):
        os.mkdir("data_med")

    with open("data_med/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        f.write(json.dumps(tokens))

    for key, ds in dataset_splits.items():
        with open(f"data_med/{key}.jsonl", "w") as f:
            for item in ds:
                newitem = {}
                newitem["input"] = (
                    f"<START_Q>{item['input']}<END_Q>"
                    f"<START_A>{item['answer_chatdoctor']}<END_A>"
                )
                f.write(json.dumps(newitem) + "\n")


if __name__ == "__main__":
    main()
