from typing import Any, Dict, Type, Union

from langchain_core.language_models import FakeListChatModel
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableLambda

from src.generate_pydantic_recipe_chain import GeneratePydancitRecipeChain, Recipe


class FakeStructuredChatModel(FakeListChatModel):
    responses: list[BaseModel]

    def with_structured_output(
        self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any
    ) -> Runnable:
        return RunnableLambda(lambda _: self.responses[0])

    @property
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


def test_invoke() -> None:
    response_recipe = Recipe(
        ingredients=["たまねぎ", "にんじん"],
        steps=["あああ", "いいい"],
    )
    llm = FakeStructuredChatModel(responses=[response_recipe])
    chain = GeneratePydancitRecipeChain(llm=llm)

    assert chain.invoke("カレー") == "response_recipe"
