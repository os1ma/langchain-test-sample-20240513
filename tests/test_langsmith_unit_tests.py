import pytest
from dotenv import load_dotenv

load_dotenv()

from langsmith import expect, traceable, unit


@traceable  # Optional
def generate_sql(user_query):
    # Replace with your SQL generation logic
    # e.g., my_llm(my_prompt.format(user_query))
    return "SELECT * FROM customers"


# @unit
# def test_sql_generation_select_all():
#     user_query = "Get all users from the customers table"
#     sql = generate_sql(user_query)
#     # LangSmith logs any exception raised by `assert` / `pytest.fail` / `raise` / etc.
#     # as a test failure
#     assert sql == "SELECT * FROM customers"


@pytest.fixture
def user_query():
    return "Get all users from the customers table"


@pytest.fixture
def expected_sql():
    return "SELECT * FROM customers"


# output_keys indicate which test arguments to save as 'outputs' in the dataset (Optional)
# Otherwise, all arguments are saved as 'inputs'
@unit(output_keys=["expected_sql"])
def test_sql_generation_with_fixture(user_query, expected_sql):
    sql = generate_sql(user_query)
    assert sql == expected_sql


@unit
@pytest.mark.parametrize(
    "user_query, expected_sql",
    [
        ("Get all users from the customers table", "SELECT * FROM customers"),
        ("Get all users from the orders table", "SELECT * FROM orders"),
    ],
)
def test_sql_generation_parametrized(user_query, expected_sql):
    sql = generate_sql(user_query)
    assert sql == expected_sql


@unit
def test_sql_generation_select_all():
    user_query = "Get all users from the customers table"
    sql = generate_sql(user_query)
    expect(sql).to_contain("customers")
