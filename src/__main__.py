import llm_sdk
import json
import argparse
from pydantic import BaseModel, model_validator, Field, ConfigDict
import os
import re


class FunctionDefinition(BaseModel):
    name: str
    arguments: list[list[str]]
    return_type: str
    description: str


class FunctionCall(BaseModel):
    prefix: str = ''
    postfix: str = """
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}

Now the real one:

"""
    og_prompt: str
    prompt: str | None = None
    name_pattern: str = re.compile(r'"name": "[^,]*\Z')
    parameter_pattern: str = re.compile(r'"parameters"\s*:\s*\{\s*"')
    complete_fn_name: str = "not_found"
    fds: list[FunctionDefinition]

    @model_validator(mode='after')
    def set_postfix(self):
        self.prefix += self.postfix

    @model_validator(mode='after')
    def space_after_prompt(self):
        breakpoint()
        if not self.og_prompt.endswith(" "):
            self.og_prompt += ' '
        return self

    @model_validator(mode='before')
    @classmethod
    def set_original_prompt(cls, data):
        if data.get("og_prompt") is None:
            data['og_prompt'] = data['prompt']
        return data

    def response(self, replace_spaces: bool = True):
        return self.prompt[len(self.og_prompt):].replace("Ġ", " ")\
               if replace_spaces else self.prompt[len(self.og_prompt):]

    def is_json(self, logit: str):
        pass

    def reprompt(self, m, og: int = 0) -> bool:
        vocab = m.get_vocab()
        input_ids = m.model.encode(self.prefix + self.prompt)
        logits = m.model.get_logits_from_input_ids(input_ids.tolist()[0])
        vocab_list: tuple[str, float, int] = [(vocab[token_id], logits[token_id], token_id)
                      for token_id in range(len(vocab))]
        # constrained decoding !
        print(self.prompt)
        # if (self.prompt[-9:] == '"name": "'):
        # fn_name constraint
        if self.name_pattern.search(self.prompt)\
           and self.complete_fn_name == "not_found":
            fnames: list[str] = [f.name for f in m.fdefinitions]
            partial_fn_name = self.prompt.split("name\": \"")[1]
            fnames_tokens: list[int] = [
                t for sub in
                [m.model.encode(fn).tolist()[0] for fn in fnames] for t in sub]
            constrained_vocab: tuple[str, float, int] = [
                logit for logit in vocab_list if logit[2] in fnames_tokens and
                any([re.search((partial_fn_name + logit[0]), fn_name)
                     for fn_name in fnames])
            ]
            if partial_fn_name in fnames:
                self.complete_fn_name = partial_fn_name
                self.prompt = self.prompt + "\","
            if self.complete_fn_name == "not_found":
                self.prompt = self.prompt + m.model.decode(
                    max(constrained_vocab, key=lambda x: x[1])[2])
        elif self.parameter_pattern.search(self.prompt):
            fd: FunctionDefinition = [fd for fd in self.fds if fd.name == self.complete_fn_name][0]
            fd_args = fd.arguments
            partial_args: list[str] = self.prompt.split("parameters\": {\"")[1]
            parsed_partial_args: list[str] = partial_args.split("\"")
            # constrained_vocab: tuple[str, float, int] = [
            #     logit for logit in vocab_list
            #     if logit[0] == fd_args[(len(parsed_partial_args) % 3) - 1][0]
            #     and logit[0] not in parsed_partial_args
            # ]
            # breakpoint()
            # if len(constrained_vocab) > 0:
            #     self.prompt = self.prompt + m.model.decode(
            #         max(constrained_vocab, key=lambda x: x[1])[2])
            # else:
            self.prompt = self.prompt + m.model.decode(
                max(vocab_list, key=lambda x: x[1])[2])
        else:
            self.prompt = self.prompt + m.model.decode(
                max(vocab_list, key=lambda x: x[1])[2])
        if (self.response().count('{') == 1
                and self.response().count('}') == 2):
            return 1
        return 0


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: llm_sdk.Small_LLM_Model = Field(
        default_factory=llm_sdk.Small_LLM_Model)
    responses: list[str] = Field(default_factory=list)
    fdefinitions: list[FunctionDefinition] = Field(default_factory=list)

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
    prefix: str = functions_definition + function_calling_tests
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
                prompt="{\n\t\"prompt\": \"" + fc['prompt'] + "\",\n\t",
                fds=fds,
                prefix=prefix
            ))
    m = Model(fdefinitions=fds)
    for fcalls in fcs:
        finished: bool = fcalls.reprompt(m)
        while (not finished):
            finished = fcalls.reprompt(m)
        m.responses.append(fcalls.prompt)
    if args.output is None:
        try:
            os.makedirs('data/output', exist_ok=True)
            with open('data/output/function_calls.json', 'w') as file:
                for r in m.responses:
                    file.write(r)
        except Exception as e:
            print(f"There is an issue with the output file!: \n{e}")
    else:
        #to fix
        try:
            os.makedirs((yes := args.output.rsplit('/', 1))[0], exist_ok=True)
            with open(yes[1], 'w') as file:
                for r in m.responses:
                    file.write(r)
        except Exception as e:
            print(f"There is an issue with the output file!: \n{e}")


if __name__ == '__main__':
    main()
