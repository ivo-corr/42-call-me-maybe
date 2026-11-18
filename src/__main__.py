"""Function calling with constrained decoding on a small LLM.

Translates natural-language requests into JSON function calls. The model
only chooses among valid function names and generates the values of the
parameters, while the JSON structure itself is forced and built in Python,
so the output is always valid JSON.
"""

import llm_sdk
import json
import argparse
from pydantic import BaseModel, model_validator, Field, ConfigDict
import os
import re
from typing import Any, cast
import numpy as np
import time


class FunctionDefinition(BaseModel):
    """Description of a function the model is allowed to call.

    Attributes:
        name: Name of the function (e.g. ``fn_add_numbers``).
        arguments: List of ``[parameter_name, parameter_type]`` pairs, in
            the order the parameters must be generated.
        return_type: Type returned by the function.
        description: Human-readable description of what it does.
    """

    name: str
    arguments: list[list[str]]
    return_type: str
    description: str


class FunctionCall(BaseModel):
    """A single user request being converted into a function call.

    Holds the few-shot prefix and the prompt, and runs the constrained
    decoding that produces the final JSON function call. After
    ``reprompt`` finishes, ``prompt`` contains the resulting JSON text.

    Attributes:
        prefix: Text prepended to the request as context for the model.
        og_prompt: Original prompt, used to slice off the generated part.
            Defaults to ``prompt`` and always ends with a space.
        prompt: Current prompt text. Replaced by the final JSON string
            once generation is done.
        name_pattern: Regex matching an unfinished ``"name"`` field.
        parameter_pattern: Regex matching the start of the ``"parameters"``
            object.
        complete_fn_name: Name of the function that was selected.
        fds: Definitions of all the available functions.
    """

    prefix: str = """
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}

Now the real one:

"""
    og_prompt: str = ''
    prompt: str | None = None
    name_pattern: re.Pattern[str] = re.compile(r'"name": "[^,]*\Z')
    parameter_pattern: re.Pattern[str] = re.compile(
        r'"parameters"\s*:\s*\{\s*"')
    complete_fn_name: str = "not_found"
    fds: list[FunctionDefinition]

    @model_validator(mode='after')
    def space_after_prompt(self) -> "FunctionCall":
        """Ensure ``og_prompt`` ends with a trailing space.

        Returns:
            The validated instance, with ``og_prompt`` normalized.
        """
        if not self.og_prompt.endswith(" "):
            self.og_prompt += ' '
        return self

    @model_validator(mode='before')
    @classmethod
    def set_original_prompt(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Default ``og_prompt`` to the value of ``prompt``.

        Args:
            data: Raw input data, before validation.

        Returns:
            The input data, with ``og_prompt`` filled in if it was missing.
        """
        if data.get("og_prompt") is None:
            data['og_prompt'] = data['prompt']
        return data

    def response(self, replace_spaces: bool = True) -> str | None:
        """Return only the text generated after the original prompt.

        Args:
            replace_spaces: If True, replace the byte-level space marker
                ``Ġ`` with a regular space.

        Returns:
            The generated text, or None if there is no prompt yet.
        """
        if self.prompt is None:
            return None
        generated = self.prompt[len(self.og_prompt):]
        return generated.replace("Ġ", " ") if replace_spaces else generated

    def reprompt(self, m: "Model",
                 constrain: list[int],
                 vocab: dict[int, str],
                 og: int = 0,
                 _cache: dict[str, Any] = {},
                 ) -> int:
        """Run constrained decoding and store the JSON call in ``prompt``.

        The function name is chosen token by token, restricted to pieces
        that keep it a prefix of a real function name. The parameter keys
        are forced and only their values are generated: strings may not
        contain control characters or backslashes, and numbers may only
        use digits, ``.`` and ``-``. The final JSON is built with
        ``json.dumps``, so it is always valid.

        Args:
            m: Model wrapper holding the LLM and the function definitions.
            constrain: Unused, kept for compatibility with the caller.
            vocab: Mapping of token id to token string.
            og: Unused, kept for compatibility with the caller.
            _cache: Mutable default argument used on purpose as a
                persistent cache. It is filled on the first call and reused
                afterwards. It is not meant to be passed by the caller.

        Returns:
            1 when the call is finished (always, as everything is done in
            a single call).
        """
        if self.prompt is None:
            return 1

        MAX_NAME, MAX_VALUE = 12, 40
        names = [f.name for f in m.fdefinitions] + ["fn_unknown_function"]

        def enc(text: str) -> list[int]:
            """Encode ``text`` into a list of token ids."""
            return cast(list[int], m.model.encode(text).tolist()[0])

        def scores(ids: list[int]) -> np.ndarray[Any, np.dtype[np.float64]]:
            """Return the next-token logits for the token ids ``ids``."""
            return np.asarray(
                m.model.get_logits_from_input_ids(ids), dtype=float)

        # one-time setup: runs on the first call only, then reused
        if not _cache:
            ids_in_names = {t for n in names for t in enc(n)}
            _cache["name_pieces"] = {
                t: m.model.decode(t) for t in ids_in_names}
            # tokens not allowed inside a string value
            # (byte-level markers: Ċ newline, ĉ tab, č carriage return)
            _cache["banned"] = np.array(
                [t for t, s in vocab.items()
                 if any(c in s for c in "Ċĉč\\")], dtype=int)
            # tokens made only of digits, '.' and '-'
            _cache["number"] = np.array(
                [t for t, s in vocab.items()
                 if s and all(c in "0123456789.-" for c in s)], dtype=int)
            # tokens that end a number
            _cache["terms"] = {enc(c)[0] for c in (",", "}")}
            _cache["quote"] = enc('"')[0]
        name_pieces: dict[int, str] = _cache["name_pieces"]
        quote_id: int = _cache["quote"]

        # recover the question from the starting prompt that main builds
        question = json.loads(
            self.prompt.rstrip().rstrip(",") + "}")["prompt"]

        # forced start of the output: costs no forward pass by itself
        ids = enc(self.prefix + '{"prompt": ' + json.dumps(question)
                  + ', "name": "')

        # 1. function name: only pieces that keep us a prefix of a real name
        name = ""
        for _ in range(MAX_NAME):
            ext = [t for t, text in name_pieces.items()
                   if any(n.startswith(name + text) for n in names)]
            if not ext:
                break
            cands = ext + ([quote_id] if name in names else [])
            s = scores(ids)
            t = max(cands, key=lambda c: s[c])
            if t == quote_id:
                break
            ids.append(t)
            name += name_pieces[t]

        # 2. parameters: keys are forced, only the values are generated
        fdef = next((f for f in m.fdefinitions if f.name == name), None)
        params: dict[str, Any] = {}
        ids += enc('", "parameters": {')
        for i, (pname, ptype) in enumerate(fdef.arguments if fdef else []):
            is_str = ptype == "string"
            ids += enc(("" if i == 0 else ", ") + json.dumps(pname) + ": "
                       + ('"' if is_str else ""))
            text = ""
            for _ in range(MAX_VALUE):
                s = scores(ids)
                if is_str:
                    s[_cache["banned"]] = -np.inf
                    t = int(np.argmax(s))
                    piece = m.model.decode(t)
                    if '"' in piece:            # closing quote: value done
                        text += piece.split('"')[0]
                        break
                else:
                    allowed = np.full_like(s, -np.inf)
                    allowed[_cache["number"]] = s[_cache["number"]]
                    if text:                    # need at least one character
                        for tid in _cache["terms"]:
                            allowed[tid] = s[tid]
                    t = int(np.argmax(allowed))
                    if t in _cache["terms"]:    # number done
                        break
                    piece = m.model.decode(t)
                ids.append(t)
                text += piece
            if is_str:
                params[pname] = text
                ids += enc('"')
            else:
                try:
                    params[pname] = (int(float(text)) if ptype == "integer"
                                     else float(text))
                except ValueError:
                    params[pname] = 0.0

        # 3. build the output in Python, so it is always valid JSON
        self.prompt = json.dumps(
            {"prompt": question, "name": name, "parameters": params},
            indent=2)
        return 1


class Model(BaseModel):
    """Wrapper around the LLM, its responses and the known functions.

    Attributes:
        model: The underlying ``llm_sdk`` small language model.
        responses: JSON function calls generated so far.
        fdefinitions: Definitions of the available functions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: llm_sdk.Small_LLM_Model = Field(
        default_factory=llm_sdk.Small_LLM_Model)
    responses: list[str] = Field(default_factory=list)
    fdefinitions: list[FunctionDefinition] = Field(default_factory=list)

    def get_vocab(self) -> dict[int, str]:
        """Load the tokenizer vocabulary from the model's vocab file.

        Returns:
            A mapping of token id to token string.

        Raises:
            SystemExit: If the vocab file cannot be read or parsed (an
                error message is printed first).
        """
        try:
            with open(self.model.get_path_to_vocab_file()) as file:
                vocab_str: str = file.read()
                raw_vocab: dict[str, int] = json.loads(vocab_str)
                vocab = {
                    int(token_id): token
                    for token, token_id in raw_vocab.items()
                }
        except Exception:
            print("Something went wrong retrieving the vocab.")
            exit()
        return vocab


def build_prefix(functions_definition: list[dict[str, Any]]) -> str:
    """Build the instruction and few-shot prefix for the model.

    The prefix lists the available functions with their signatures and
    descriptions, followed by examples of correct requests and outputs.

    Args:
        functions_definition: Raw function definitions, as loaded from the
            JSON file. Each needs ``name``, ``description`` and a
            ``parameters`` dict mapping names to ``{"type": ...}``.

    Returns:
        The prefix text, ending right before the user's request.
    """
    lines = []
    for fn in functions_definition:
        params = fn["parameters"]
        sig = ", ".join(f"{k}: {v['type']}" for k, v in params.items())
        n = len(params)
        lines.append(
            f"- {fn['name']}: takes {n} parameter{'s' if n != 1 else ''} "
            f"({sig}). {fn['description']}"
        )

    return (
        "You convert a user request into a JSON function call.\n"
        "Reply with ONE JSON object and nothing else.\n"
        'Pick exactly one function from the list. '
        'Copy the request into "prompt".\n'
        "Use EXACTLY the parameters listed for that function: "
        "no extra, no missing, same names.\n"
        "Numbers must be written as floats (2.0, not 2).\n\n"
        "Functions:\n" + "\n".join(lines) + "\n\n"
        "Examples:\n\n"
        "Request: What is 4 times 5?\n"
        '{"prompt": "What is 4 times 5?", "name": "fn_multiply", '
        '"parameters": {"x": 4.0, "y": 5.0}}\n\n'
        "Request: Make 'abc' uppercase\n"
        '{"prompt": "Make \'abc\' uppercase", "name": "fn_to_upper", '
        '"parameters": {"text": "abc"}}\n\n'
        "Request: What's the weather in Paris tomorrow?\n"
        '{"prompt": "What\'s the weather in Paris tomorrow?", '
        '"name": "fn_unknown_function", "parameters": {}}\n\n'
        "Request: "
    )


def main() -> None:
    """Run the function calling pipeline from the command line.

    Reads the function definitions and the test prompts, generates a
    function call for each prompt with constrained decoding, prints the
    total time taken and writes all the calls to a JSON file.

    Command line arguments:
        --functions_definition: Path to the functions definition JSON
            (default: ``data/input/functions_definition.json``).
        --input: Path to the test prompts JSON
            (default: ``data/input/function_calling_tests.json``).
        --output: Path of the output JSON
            (default: ``data/output/function_calls.json``).

    Raises:
        SystemExit: If an input file cannot be read.
    """
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
                functions_definition = file.read()
        if args.input is None:
            with open("data/input/function_calling_tests.json") as file:
                function_calling_tests: str = file.read()
        else:
            with open(args.input) as file:
                function_calling_tests = file.read()
    except Exception as e:
        print(f"Something went wrong: {e}")
        exit()

    fd_json: list[dict[str, Any]] = json.loads(functions_definition)
    fc_json: list[dict[str, Any]] = json.loads(function_calling_tests)

    fds: list[FunctionDefinition] = []
    fcs: list[FunctionCall] = []

    start_time = time.perf_counter()
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
                prefix=build_prefix(fd_json),
                prompt="{\n\t\"prompt\": " + json.dumps(fc['prompt'])
                + ",\n\t",
                fds=fds
                # prompt=fc['prompt']
            ))
    m = Model(fdefinitions=fds)
    vocab: dict[int, str] = m.get_vocab()
    for fcalls in fcs:
        finished: int = fcalls.reprompt(m, [1, 0], vocab)
        while (not finished):
            finished = fcalls.reprompt(m, [1, 0], vocab)
        m.responses.append(fcalls.prompt) if fcalls.prompt is not None\
            else None
    finish_time = time.perf_counter()
    print("====\ntotal time: " +
          str((finish_time - start_time)/60) + " minutes\n====")

    output_path: str = args.output if args.output is not None \
        else 'data/output/function_calls.json'
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as file:
            file.write("[\n"
                       + ",\n".join(r.strip() for r in m.responses)
                       + "\n]\n")
    except Exception as e:
        print(f"There is an issue with the output file!: \n{e}")


if __name__ == '__main__':
    main()
