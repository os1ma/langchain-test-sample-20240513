from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage

from src.generate_recipe_chain import GenerateRecipeChain


def test_invoke() -> None:
    response = """それではカレーのレシピを生成します。

```
## 材料

- たまねぎ
- にんじん

## 手順

1. あああ
2. いいい
```
"""

    responses: list[BaseMessage] = [AIMessage(content=response)]
    llm = FakeMessagesListChatModel(responses=responses)
    chain = GenerateRecipeChain(llm=llm)
    actual = chain.invoke(dish="カレー")

    expected = """## 材料

- たまねぎ
- にんじん

## 手順

1. あああ
2. いいい
"""
    assert actual == expected
