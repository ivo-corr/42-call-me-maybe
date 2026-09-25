import llm_sdk
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
    print(model.encode("Diana"))
    print(args.functions_definition is not None)
    print(args.input is not None)


if __name__ == '__main__':
    main()
