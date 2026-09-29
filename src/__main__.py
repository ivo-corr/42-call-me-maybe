import llm_sdk
import json
import argparse
from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    name: str
    arguments: list[list[str]]
    return_type: str
    description: str


class FunctionCall(BaseModel):
    prompt: str

    def reprompt(logit):
        prompt = prompt + logit

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
        else:
            with open(args.functions_definition) as file:
                functions_definition: str = file.read()
        if args.input is None:
            with open("data/input/function_calling_tests.json") as file:
                function_calling_tests: str = file.read()
        else:
            with open(args.input) as file:
                function_calling_tests: str = file.read()
    except Exception as e:
        print(f"Something went wrong: {e}")
        exit()
    
    fd_json: list[dict] = json.loads(functions_definition)
    fc_json: list[dict] = json.loads(function_calling_tests)

    fds: list[FunctionDefinition] = []
    for fd in fd_json:
        fds.append(FunctionDefinition(
            name=fd['name'],
            arguments=[[param, fd['parameters'][param]['type']]
                       for param in fd['parameters']],
            return_type=fd['returns']['type'],
            description=fd['description']
        ))
    model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()
    with open(model.get_path_to_vocab_file()) as file:
        vocab_str: str = file.read()
        vocab: list[dict] = json.loads(vocab_str)
        vocab = {int(token_id): token for token, token_id in vocab.items()}
    prompt: str = 'What is 2 + 2?'
    while (True):
        input_ids = model.encode(prompt)
        logits = model.get_logits_from_input_ids(input_ids.tolist()[0])
        vocab_list = [(vocab[token_id], logits[token_id])
                      for token_id in range(len(vocab))]
        print(prompt.replace("Ġ", " "))
        breakpoint()
        prompt = prompt + max(vocab_list, key=lambda x: x[1])[0]


if __name__ == '__main__':
    main()
