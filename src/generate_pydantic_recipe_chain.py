from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()


class Recipe(BaseModel):
    ingredients: list[str]
    steps: list[str]


_PROMPT_TEMPLATE = """料理のレシピを教えてください。

料理名: {dish}
"""


class GeneratePydancitRecipeChain:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def invoke(self, dish: str) -> Recipe:
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)

        chain: Runnable[Any, Any] = (
            {"dish": RunnablePassthrough()}
            | prompt
            | self.llm.with_structured_output(schema=Recipe)
        )

        return chain.invoke(dish)


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    chain = GeneratePydancitRecipeChain(llm=llm)
    ai_message = chain.invoke("カレー")
    print(ai_message)
