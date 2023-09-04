[![Guanaco Banner](https://imgur.com/vUn3OXJ.png)](https://www.chai-research.com/competition.html)

[![PyPI version](https://badge.fury.io/py/chai-guanaco.svg)](https://badge.fury.io/py/chai-guanaco)
[![first-timers-only Friendly](https://img.shields.io/badge/first--timers--only-friendly-blue.svg)](http://www.firsttimersonly.com/)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)


# Hosted By
<a href="https://www.chai-research.com/"><img src="https://imgur.com/odX7Jz4.png" alt="Chai Logo" height="90"/></a>

[Chai Guanaco](https://www.chai-research.com/competition.html) is part of the Chai Guanaco Competition, accelerating community AGI.

It's the world's first open community challenge with real-user evaluations. You models will be directly deployed on the [Chai App](http://tosto.re/chaiapp) where our over 500K daily active users will be providing live feedback. Get to top of the leaderboard and share the $1 million cash prize!

# 🚀 Quick Start

[Chai Guanaco Jupyter Notebook Quickstart](https://colab.research.google.com/drive/1FyCamT6icUo5Wlt6qqogHbyREHQQkAY8?usp=sharing)


## The Guanaco Guide

🥇 **Evaluation & Prizes:** Depending on the phase of the competition, a suite of user-level evaluation metrics will be used (i.e. thumbs up / thumbs down rate). Your model will be ranked in real-time compared with other models, you can view the leaderboard at anytime with the pip package

🕵️ **Real-time user feedback:** After your model is deployed, it will go through an safety + integrity checker, once passed, it will be deployed directly to our users who will provide written feedback that you can view via the pip package.

🤖 **Model requirements:** Currently, we support *any* models based off [LLaMA 2](https://huggingface.co/docs/transformers/model_doc/llama2) (i.e. 7/13B parameters with LLaMA tokenizer). All you need to do is push your model directly to huggingface. Support for more model types is coming soon!

⚙️ **Sampling parameters:** During submission, we allow for custom model generation parameters such as temperature. Once your model is deployed on our platform, it will be using the parameters you've provided to generate chat completions.

📚 **Rules:** By default, 1 developer key per person, each key can deploy 1 model to users at a time. Message in discord if you would like the limit bumped up 😀


## How Does It Work?

-   The `chai_guanaco` pip package provides a way to easily submit your language model, all you need to do is ensure it is on HuggingFace 🤗
-   We will automatically **Tritonize** your model for fast inference and host it in our internal GPU cluster 🚀
-   Once deployed, Chai users on our platform who enter the **arena mode** will be rating your model directly, providing you with both quantatitive and verbal feedback 📈
-   Both the public leaderboard and **user feedback** for your model can be directly downloaded via the `chai_guanaco` package 🧠
-   Cash prizes will be allocated according to your position in the leaderboard 💰

[![Chai Pipeline](https://imgur.com/LtMWOAq.png)](https://www.chai-research.com/competition.html)

## Getting Started

#### **Getting Developer Key**

Join the [competition discord](https://discord.gg/7mXdjAkw2s), introduce yourself and ask for a developer key. Login-based authentication is coming next 🤗


#### **Installation**

Use [pip](https://github.com/pypa/pip) to install the Chai Guanaco package

```sh
pip install chai-guanaco
```

For one-off authentication run the following in your terminal:

```sh
chai-guanaco login
```

And pass in your developer key when prompted, you can always logout using `chai-guanaco logout`.

#### **Model Submission**

Upload any Llama based language model *with a tokenizer* to huggingface, i.e. [ChaiML/phase2_winner_13b2](https://huggingface.co/ChaiML/phase2_winner_13b2).
Read [this guide](https://huggingface.co/docs/transformers/model_sharing) if you are unsure.

To submit model simply run:

```python
import chai_guanaco as chai

model_url = "ChaiML/phase2_winner_13b2" # Your model URL

generation_parameters = {
	'temperature': 0.8,
	'top_p': 0.2,
	"top_k": 40,
	"stopping_words": ['\n'],
	"presence_penalty": 0.,
	"frequency_penalty": 0.,
	}

submission_parameters = {
    'model_repo': model_url,
    'generation_params': generation_parameters,
    'model_name': 'my-awesome-model',
    }

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
```

This will display an animation while your model is being deployed, a typical deployment takes approximately 10 minutes.
Note the `model_name` parameter is used for show-casing your model on the leaderboard and it should help you with identifying your model

- For more details on configuration and optimization of generation parameters, please check [Chai Prize Generation Parameters](https://wild-chatter-b52.notion.site/Chai-Prize-Generation-Parameters-bf6b64875dc4443986019e20fdbdc2bd).

- To submit your model with custom formatting, you can create your own `PromptFormatter`.
For more details and examples, please check [Chai Prize Prompt Format](https://wild-chatter-b52.notion.site/Chai-Prize-Prompt-Format-ec7986c40025493488a2f18e91c8cac9).


#### **Chat With Your Model Submission**

Once your model is deployed, you can verify its behaviour and raw input by running:

```python
chatbot = chai.SubmissionChatbot(submission_id)
chatbot.chat('nerd_girl', show_model_input=False)
```

Here you can have a dialog with one of the bots we have provided. To quit the chat, simply enter "exit". Note that, in order to prevent spamming, each model submission is limited to 1000 chat messages from the Chai Guanaco pip package.

You can get a list of avaliable bots by running:

```python
chatbot.show_avaliable_bots()
```

Finally, to enter a chat session that prints out the raw input that was fed into your model at each turn of the conversation, you can run:

```python
chatbot.chat('nerd_girl', show_model_input=True)
```

#### **Getting User Feedback**

Once your model has been submitted, it is automatically deployed to the Chai Platform where real-life users will evaluate your model performance. To view their feedback, run:

```python
model_feedback = chai.get_feedback(submission_id)
model_feedback.sample()
```

Which will print out one of the users' conversation, together with meta information associated with the conversation (i.e. rating and user feedback).

To get all the feedback for your model, run...

```python
df = model_feedback.df
print(df)
```

This outputs a Pandas `DataFrame`, where each row corresponds to a user conversation with your model, together with their feedback.

#### **Getting Live Leaderboard**

To view the public leaderboard used to determine prizes (which only shows the best model submitted by each developer):
```python
df = chai.display_leaderboard(detailed=False)
```

To see how your model performs against other models, run:
```python
df = chai.display_leaderboard(detailed=True)
```
which prints out the current leaderboard according to the most recent competition metrics, you can also access raw leaderboard is dumped to `df`

#### **Re-Submitting Models**

Because it is a competition, you are allowed to test a single model at any given time. However, you can deactivate a model and submit a new one. To do this, simply run:

```python
chai.deactivate_model(submission_id)
```
Which will deactive your model, don't worry, all the model feedback will still be saved, it just means the model will no longer be exposed to users. You can then re-submit by repeating the model submission step.

#### **Retrieve Your Model Submission IDs**

In case you have forgotten your submission ids / want to view all past submissions, run:

```python
submission_ids = chai.get_my_submissions()
print(submission_ids)
```
Here you will see all your model submission_ids along with their status, which is either `failed`, `inactive` or `deployed`.

#### **Advanced Usage**
- This package caches various data, such as your developer key, in the folder `~/.chai-guanaco`. To change this, you can set the environment variable `GUANACO_DATA_DIR` to point to a different folder. You may need to re-run `chai-guanaco login` to update the cached developer key.
- You can also access the raw feedback data by running
	```python
	model_feedback = chai.get_feedback(submission_id)
	raw_data = model_feedback.raw_data
	```


## Resources
|                                                                        |                                                                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------|
| 📒 [Fine tuning guide](https://huggingface.co/docs/transformers/training) | Guide on language model finetuning                                                           |
| 💾 [Datasets](https://dataset-ideas.tiiny.site/) | Curated list of open-sourced datasets to get started with finetuning                                                  |
| 💖 [Chai Prize Discord](https://discord.gg/7mXdjAkw2s)                   | Our Chai Prize Competition discord                                                          |
|🚀 [Deepspeed Guide](https://huggingface.co/docs/transformers/main_classes/deepspeed)     | Guide for training with Deepspeed (faster training without GPU bottleneck)    |
|💬 [Example Conversations](https://huggingface.co/datasets/ChaiML/100_example_conversations)     | Here you can find 100 example conversations from the Chai Platform     |
| ⚒️ [Build with us](https://boards.greenhouse.io/nexus/jobs/5319721003)| If you think what we are building is cool, join us!|


# Sponsored By

<a href="https://www.coreweave.com/"><img src="https://imgur.com/oJyuH8q.png" alt="Coreweave Logo" height="70"/></a>
