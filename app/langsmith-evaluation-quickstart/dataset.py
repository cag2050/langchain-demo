from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def main():
    client = Client()

    # Programmatically create a dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name="Sample Dataset",
        description="Sample Dataset",
    )

    # Create examples
    examples = [
        {
            "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
            "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."}
        },
        {
            "inputs": {"question": "What is Earth's lowest point?"},
            "outputs": {"answer": "Earth's lowest point is The Dead Sea."}
        },
    ]

    # Add examples to the dataset
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print("Created dataset:", dataset.name)


if __name__ == "__main__":
    main()
