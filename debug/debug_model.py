from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from configuration_chatglm import ChatGLMConfig
from datasets import load_dataset
import os
data_dir = os.getenv("my_data_dir")


def main():

    # 准本数据
    data_files = {}
    data_files["train"] = '../ptuning/AdvertiseGen/train.json'
    data_files["validation"] = '../ptuning/AdvertiseGen/dev.json'
    extension = 'json'

    raw_datasets = load_dataset(extension,
                                data_files=data_files,
                                cache_dir=None,
                                use_auth_token=None)

    print(raw_datasets)

    vocab_file = os.path.join(data_dir, 'pretrained', 'THUDM', 'chatglm2-6b', 'tokenizer.model')
    tokenizer = ChatGLMTokenizer(vocab_file=vocab_file)

    # 传入数据相关参数
    max_source_length = 64
    max_target_length = 128
    prompt_column = "content"
    response_column = "summary"
    history_column = None
    prefix = ''  # 对应的 source_prefix
    ignore_pad_token_for_loss = True
    preprocessing_num_workers = 1
    overwrite_cache = False  # TODO ?

    def preprocess_function_train(examples):
        max_seq_length = max_source_length + max_target_length + 1

        model_inputs = {  # datasets map 返回要求
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)  # 就是将问题与历史信息拼接起来，作为提示词。

                prompt = prefix + prompt  # 这里的前缀也起到了提示作用，这里没用上就是''
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,  # 提示词的ids
                                         max_length=max_source_length)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,  # 答案的ids
                                         max_length=max_target_length)

                context_length = len(a_ids)  # 上下文长度
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]  # 提示词 + 答案 + 结束词
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]  # 填充 + 答案 + 结束词

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len  # input_ids 对齐到最大长度
                labels = labels + [tokenizer.pad_token_id] * pad_len  # labels 对齐到最大长度
                if ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]  # labels中pad的词忽略掉，不计算loss

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(preprocess_function_train,
                                      batched=True,
                                      num_proc=preprocessing_num_workers,
                                      remove_columns=train_dataset.column_names,
                                      load_from_cache_file=not overwrite_cache,
                                      desc="Running tokenizer on train dataset")

    def print_dataset_example(example):
        print("input_ids=", example["input_ids"], sep='\n', end='\n\n')
        print("inputs=", tokenizer.decode(example["input_ids"]), sep='\n', end='\n\n')
        print("label_ids=", example["labels"], sep='\n', end='\n\n')
        print("labels=", tokenizer.decode(example["labels"]), sep='\n', end='\n\n')

    print_dataset_example(train_dataset[0])


    config = ChatGLMConfig(num_layers=2,
                           num_attention_heads=2)
    config.original_rope = True # 复制配置文件中的
    config.use_cache = True # 复制配置文件中的
    model = ChatGLMForConditionalGeneration(config=config)


    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(  # TODO 这个需要熟悉一下
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1, 
                                  collate_fn=data_collator)
    
    # 可以调试数据输入，到模型输出了
    for feature in train_dataloader:
        output = model.forward(**feature)
        print(output)
        break

    

    ...


if __name__ == "__main__":
    main()
    ...
