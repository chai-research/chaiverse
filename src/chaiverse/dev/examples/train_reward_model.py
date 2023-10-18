from chaiverse.dev.dataset import DatasetLoader, RewardDatasetBuilder
from chaiverse.dev.tokenizer import GPT2Tokenizer
from chaiverse.dev.model.reward_model import RewardClassificationTrainer


if __name__ == '__main__':
    # load data
    data_path = 'ChaiML/20231012_chai_prize_reward_model_data'
    data_loader = DatasetLoader(
            hf_path=data_path,
            data_samples=1000,
            validation_split_size=0.1,
            shuffle=True,
            )
    df = data_loader.load()
    print(df)

    # process data
    tokenize_loader = GPT2Tokenizer(
            padding_side='right',
            truncation_side='left',
            )
    data_builder = RewardDatasetBuilder(
            tokenize_loader=tokenize_loader,
            block_size=1024,
            )
    data = data_builder.generate(df, n_jobs=10)
    print(data)

    # train model
    model = RewardClassificationTrainer(
            model_name='gpt2',
            tokenize_loader=tokenize_loader,
            output_dir='test_reward_model',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            eval_strategy='steps',
            eval_steps=50,
            )
    print(model.training_config)
    model.fit(data)
    model_path = 'ChaiML/chai_reward_model_gpt2_test'
    model.push_to_hub(model_path, private=True)
