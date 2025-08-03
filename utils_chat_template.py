from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

import utils_structured_output

###########################################
###########################################

# Create Chat Template for Classification

# Use your Pydantic model
parser = PydanticOutputParser(pydantic_object=utils_structured_output.DocClassification)

# Auto-generate format instructions for the LLM
format_instructions = parser.get_format_instructions()

print(format_instructions)

def create_classification_prompt(document_text: str):
    """
    Create a structured classification prompt for the LLM using the Pydantic schema.
    
    Args:
        document_text (str): The raw text of the document to classify.
    
    Returns:
        str: The final prompt ready to send to the model.
    """
    system_msg = (
        "You are Qwen, created by Alibaba Cloud. "
        "Your task is to classify the document according to the provided schema. "
        "Return the result as valid JSON."
    )

    user_msg = (
        f"The document you need to analyze:\n\n{document_text}\n\n"
        f"{format_instructions}"
    )

    # Build the chat template with system + user messages
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "{doc}")
    ])

    return prompt_template.format(doc=user_msg)

###########################################
###########################################

# Create Chat Template for Named-Entity Extraction

# Use your Pydantic model
parser_investment = PydanticOutputParser(pydantic_object=utils_structured_output.Investment)
parser_personalAccount = PydanticOutputParser(pydantic_object=utils_structured_output.PersonalAccount)
parser_garnishment = PydanticOutputParser(pydantic_object=utils_structured_output.Garnishment)
parser_credit = PydanticOutputParser(pydantic_object=utils_structured_output.Credit)

# Auto-generate format instructions for the LLM
format_instructions_investment = parser_investment.get_format_instructions()
format_instructions_personalAccount = parser_personalAccount.get_format_instructions()
format_instructions_garnishment = parser_garnishment.get_format_instructions()
format_instructions_credit = parser_credit.get_format_instructions()

format_instruction_dict = {'investment':format_instructions_investment,
                           'personal_account': format_instructions_personalAccount,
                           'garnishment': format_instructions_garnishment,
                           'credit': format_instructions_credit}

pydantic_dict = {'investment': utils_structured_output.Investment,
                'personal_account': utils_structured_output.PersonalAccount,
                'garnishment': utils_structured_output.Garnishment,
                'credit': utils_structured_output.Credit}


def create_ner_prompt(document_text: str, doc_class: str, language: str):
    """
    Create a structured Named-Entity Extraction prompt for the LLM using the Pydantic schema.
    
    Args:
        document_text (str): The raw text of the document to classify.
        doc_class (str): The document class
        language (str): The language of the document
    
    Returns:
        str: The final prompt ready to send to the model.
    """
    system_msg = (
        "You are Qwen, created by Alibaba Cloud. "
        "Your task is to extract proper entities from the document according to the provided schema. "
        f"Return the result as valid JSON."
    )

    user_msg = (
        f"The document you need to analyze:\n\n{document_text} \n\n"
        f"{format_instruction_dict[doc_class]}"
    )

    # Build the chat template with system + user messages
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", "{doc}")
    ])

    return prompt_template.format(doc=user_msg)

###########################################
###########################################