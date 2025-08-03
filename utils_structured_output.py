from enum import Enum
from datetime import datetime
from langchain_community.llms import Outlines

from pydantic import BaseModel
from pydantic import BaseModel, Field, validator

###########################################
###########################################

# Enum for document classes
class DocumentClasses(str, Enum):
    investment = "investment"
    personal_account = "personal_account"
    garnishment = "garnishment"
    credit = "credit"

# Enum for languages
class DocumentLanguage(str, Enum):
    english = "English"
    french = "French"
    italian = "Italian"
    spanish = "Spanish"
    german = "German"

# Description map for document classes
DOCUMENT_CLASS_DESCRIPTIONS = {
    DocumentClasses.investment: "Investment-related documents (e.g., portfolio statements, fund details)",
    DocumentClasses.personal_account: "Personal account documents such as statements, account closures, or balance details",
    DocumentClasses.garnishment: "Legal notices regarding wage garnishment or account seizure",
    DocumentClasses.credit: "Credit or loan-related documents, including credit card statements and loan agreements"
}

# Pydantic model for classification
class DocClassification(BaseModel):
    doc_class: DocumentClasses = Field(
        description="Class of the document. Possible values:\n"
                    f"- {DocumentClasses.investment.value}: {DOCUMENT_CLASS_DESCRIPTIONS[DocumentClasses.investment]}\n"
                    f"- {DocumentClasses.personal_account.value}: {DOCUMENT_CLASS_DESCRIPTIONS[DocumentClasses.personal_account]}\n"
                    f"- {DocumentClasses.garnishment.value}: {DOCUMENT_CLASS_DESCRIPTIONS[DocumentClasses.garnishment]}\n"
                    f"- {DocumentClasses.credit.value}: {DOCUMENT_CLASS_DESCRIPTIONS[DocumentClasses.credit]}"
    )
    language: DocumentLanguage = Field(
        description="Language of the document. Possible values: English, French, Italian, Spanish, German"
    )

###########################################
###########################################

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Investment(BaseModel):
    portfolio_id: str = Field(description="Unique identifier for the investment portfolio")
    portfolio_value: float = Field(description="Current total value of all investments in the portfolio")
    asset_number: str = Field(description="Number of different asset types of the portfolio")
    risk_profile: RiskLevel = Field(description="Assessment of the portfolio's risk level")

###########################################
###########################################

class AccountType(str, Enum):
    checking = "checking"
    savings = "savings"
    money_market = "money_market"
    certificate_of_deposit = "certificate_of_deposit"
    credit = "credit"
    loan = "loan"

class PersonalAccount(BaseModel):
    account_number: str = Field(description="Partially masked bank account number")
    account_type: AccountType = Field(description="Type of account")
    transaction_number: str = Field(description="Number of financial activities affecting the account during the statement period")


###########################################
###########################################

class Garnishment(BaseModel):
    debtor_name: str = Field(description="Name of the individual whose assets are being garnished")
    creditor_name: str = Field(description="Name of the entity (person/organization) to whom the debt is owed")
    effective_date: datetime = Field(description="Date when the garnishment takes effect. Format: dd/mm/yyyy")

    @validator("effective_date", pre=True)
    def parse_effective_date(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%d/%m/%Y")
            except ValueError:
                raise ValueError("effective_date must be in format dd/mm/yyyy")
        return v

    class Config:
        # ✅ Serialization as dd/mm/yyyy
        json_encoders = {
            datetime: lambda v: v.strftime("%d/%m/%Y")
        }


###########################################
###########################################

class Credit(BaseModel):
    card_number: str = Field(description="Partially masked credit card number")
    credit_limit: float = Field(description="Maximum amount of credit extended to the customer")
    interest_rate: float = Field(description="Annual percentage rate applied to outstanding balances")    