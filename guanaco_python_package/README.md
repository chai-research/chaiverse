# What is chai-guanaco?
chai-guanaco allows AI engineers to rapidly deploy large language models to the
Chai app: the world's largest platform for conversational AI.

With this package you can deploy your model for free to millions of Chai users and
immediately start receiving feedback!

chai-guanaco is also the official interface for submitting a model to the Guanaco
competition: the world's first competition that pits LLM developers against each
other, using real user feedback to determine the winner. For more information
regarding this competition, and the prize-money, click
[here](https://www.chai-research.com/competition.html).

# Quickstart

To install chai-gunaco, simply run `pip install chai-guanaco`.

The following is a quick example demonstrating how you can submit your first model:

```python3
import time

from chai_guanaco import submit

model_submission = {
	"model_repo": "ChaiML/chai-llm-v4", # HuggingFace repo hosting your
	model
	"developer_uid": "XXX",
	"generation_params": {
		"temperature": 1.0,
		"top_p": 1.0,
		"top_k": 40,
		"repetition_penalty": 1.0
	}
}

response = submit.submit_model(model_submission, developer_key = "XXX")
submission_id = response["submission_id"]

while True:
	model_info = submit.get_model_info(submission_id, developer_key = "XXX")
	status = model_info["status"]
	if status != "pending":
		print(f"Model submission finished with status: {status}!")
		break
	time.sleep(10)
```

At this point, your model is now being served to users! To start viewing user
feedback, you can simply execute the following:

```python3
from chai_guanaco import feedback

feedback = feedback.get_feedback(submission_id, developer_key="XXX")
```

# API Documentation

Full documentation describing how to interface with the Guanaco API is coming
soon!

# FAQ

## My submission failed! What should I do?

You can inspect `model_info["error"]` to view the error associated with your
failed submission. Feel free to reach out to us on our Discord channel
[here](https://www.chai-research.com/competition.html) for help in debugging
your submission.

## Why are you letting me deploy my model for free to the Chai app?

We believe that by 2025, everyone will have an AI best-friend. We also believe
that this vision can only be realised by providing open-source LLM developers
with the feedback they need to quickly iterate and build better models.

## What is a guanaco anyway?

[It's like a llama!](https://en.wikipedia.org/wiki/Guanaco)
