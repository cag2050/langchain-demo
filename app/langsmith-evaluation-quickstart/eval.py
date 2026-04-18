from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

load_dotenv()

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)


# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
def target(inputs: dict) -> dict:
    # 使用 LangChain 的 invoke 方法调用模型
    response = model.invoke(
        [
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ]
    )
    # response 是一个 AIMessage 对象，其内容在 .content 属性中
    return {"answer": response.content.strip()}


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="deepseek:deepseek-chat",
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )


def main():
    client = Client()
    experiment_results = client.evaluate(
        target,
        data="Sample Dataset",  # 注意：这里是：dataset_name
        evaluators=[correctness_evaluator],
        experiment_prefix="first-eval-in-langsmith",
        max_concurrency=2,
    )
    print(experiment_results)


if __name__ == "__main__":
    main()
