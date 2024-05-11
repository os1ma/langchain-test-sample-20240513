import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

_PROMPT_TEMPLATE = """料理のレシピを教えてください。

料理名: {dish}

以下のマークダウン形式で出力してください。

```
## 材料

- ...
- ...

## 手順

1. ...
2. ...
```
"""


def extract_codeblock(text: str) -> str:
    """
    コードブロックを抽出します。
    コードブロックが存在しない場合はそのままの文字列を返します。
    """

    # 「```」で囲まれた部分を抽出
    match = re.search(r"```([^\n]*)\n(.*)```", text, re.DOTALL)
    if match:
        return match.group(2)
    else:
        return text


class GenerateMarkdownRecipeChain:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def invoke(self, dish: str) -> str:
        prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
        chain: Runnable[Any, str] = (
            {"dish": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
            | extract_codeblock
        )

        return chain.invoke(dish)


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    chain = GenerateMarkdownRecipeChain(llm=llm)
    ai_message = chain.invoke("カレー")
    print(ai_message)
