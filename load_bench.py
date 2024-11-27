import datasets

dataset = datasets.load_dataset("nguha/legalbench", "abercrombie")
print(dataset["train"].to_pandas())
print(dataset["test"].to_pandas())