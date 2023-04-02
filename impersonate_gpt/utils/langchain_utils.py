from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


def get_actual_prompt(chain: BaseCombineDocumentsChain, question: str):
    """Retrieve the actual prompt sent to the LLM API. Including all templating."""

    inputs = chain.combine_documents_chain._get_inputs(chain._get_docs(question), question=question)
    prompts, _ = chain.combine_documents_chain.llm_chain.prep_prompts([inputs])

    # Currently only supports single prompts
    return prompts[0].text
