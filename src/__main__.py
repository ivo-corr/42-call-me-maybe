import llm_sdk
import json
import argparse


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--functions_definition", type=str)
    argparser.add_argument("--input", type=str)
    argparser.add_argument("--output", type=str)
    args = argparser.parse_args()

    try:
        if args.functions_definition is None:
            with open("data/input/functions_definition.json") as file:
                functions_definition: str = file.read()
        if args.input is None:
            with open("data/input/function_calling_tests.json") as file:
                function_calling_tests: str = file.read()
    except Exception as e:
        print(f"Something went wrong: {e}")
        exit()
    model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()
    with open(model.get_path_to_vocab_file()) as file:
        vocab_str = file.read()
        vocab = json.loads(vocab_str)
        vocab = {int(token_id): token for token, token_id in vocab.items()}
    # print(input_ids.tolist())
    # print(type(input_ids))
    prompt: str = 'What is 2 + 2?'
    while (True):
        input_ids = model.encode(prompt)
        logits = model.get_logits_from_input_ids(input_ids.tolist()[0])
        vocab_list = [(vocab[token_id], logits[token_id])
                      for token_id in range(len(vocab))]
        print(prompt.replace("Ġ", " "))
        prompt = prompt + max(vocab_list, key=lambda x: x[1])[0]
    # for token_id in range(len(vocab)):
    #     token = vocab[token_id]
    #     print(token, token_id, logits[token_id])
    # print(vocab_list)


if __name__ == '__main__':
    main()
