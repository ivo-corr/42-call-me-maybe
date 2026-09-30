import llm_sdk
import json
import argparse
from pydantic import BaseModel, model_validator


class FunctionDefinition(BaseModel):
    name: str
    arguments: list[list[str]]
    return_type: str
    description: str


class FunctionCall(BaseModel):
    og_prompt: str
    prompt: str | None = None

    @model_validator(mode='after')
    def space_after_prompt(self):
        if not self.og_prompt.endswith(" "):
            self.og_prompt += ' '
        return self

    @model_validator(mode='before')
    @classmethod
    def set_original_prompt(cls, data):
        if data.get("og_prompt") is None:
            data['og_prompt'] = data['prompt']
        return data

    def response(self):
        return self.prompt[len(self.og_prompt):]

    def is_json_compliant(self, logit: str):
        pass

    def reprompt(self, m, og: int = 0):
        vocab = m.get_vocab()
        input_ids = m.model.encode(self.prompt)
        logits = m.model.get_logits_from_input_ids(input_ids.tolist()[0])
        vocab_list = [(vocab[token_id], logits[token_id])
                      for token_id in range(len(vocab))]
        print(self.prompt.replace("Ġ", " "))
        self.prompt = self.prompt + max(vocab_list, key=lambda x: x[1])[0]


class Model():
    def __init__(self):
        self.model: llm_sdk.Small_LLM_Model = llm_sdk.Small_LLM_Model()
    
    def get_vocab(self):
        try:
            with open(self.model.get_path_to_vocab_file()) as file:
                vocab_str: str = file.read()
                vocab: list[dict] = json.loads(vocab_str)
                vocab = {
                    int(token_id): token for token, token_id in vocab.items()}
        except Exception:
            print("Something went wrong retrieving the vocab.")
            exit()
        return vocab


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
    fcs: list[FunctionCall] = []

    for fd in fd_json:
        fds.append(FunctionDefinition(
            name=fd['name'],
            arguments=[[param, fd['parameters'][param]['type']]
                       for param in fd['parameters']],
            return_type=fd['returns']['type'],
            description=fd['description']
        ))
    for fc in fc_json:
        fcs.append(
            FunctionCall(
                json_prompt=str(fc),
                prompt=fc['prompt']
            ))

    m = Model()

    for fcalls in fcs:
        while (True):
            fcalls.reprompt(m)


if __name__ == '__main__':
    main()
