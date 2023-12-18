import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from configuration_chatglm import ChatGLMConfig
from modeling_chatglm import ChatGLMModel
from tokenization_chatglm import ChatGLMTokenizer
import torch

data_dir = os.getenv("my_data_dir")
model_path = os.path.normpath(os.path.join(data_dir, 'pretrained', 'THUDM/chatglm2-6b'))
print(model_path)

def debug_tokenizer():
    """
    tokenizer 在 tokenization_chatglm.py 中
    """
    # tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    tokenizer = ChatGLMTokenizer(vocab_file=os.path.join(model_path, 'tokenizer.model'))
    
    txt = '同志们好！'
    tokens = tokenizer.tokenize(txt) # 分词
    print(tokens)

    token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
    print(token_ids)

    input_ids = tokenizer.encode(text=txt)
    print(input_ids)
    print(tokenizer.convert_ids_to_tokens(input_ids))
    ...

def debug_model_config():
    """
    配置在 configuration_chatglm.py 中。
    调试chatglm2-6b的模型配置
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("AutoConfig.from_pretrained = ", config, sep='\n', end='\n\n')

    config = ChatGLMConfig.from_pretrained(model_path)
    print("ChatGLMConfig.from_pretrained = ", config, sep='\n', end='\n\n')

    config = ChatGLMConfig()
    print("ChatGLMConfig() = ", config, sep='\n', end='\n\n')


def debug_model():
    """
    模型在 modeling_chatglm.py 中。 
    """
    config = ChatGLMConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float32 # 在cpu上调试，所以这里改为float32
    
    # 为了能跑起来，降低模型复杂度
    config.ffn_hidden_size = 1024
    config.num_attention_heads = 2
    config.num_layers = 3

    model = ChatGLMModel(config=config)
    tokenizer = ChatGLMTokenizer(vocab_file=os.path.join(model_path, 'tokenizer.model'))

    txt = '同志们好！'
    input_ids = tokenizer.encode(txt, return_tensors='pt')
    
    result = model.forward(input_ids)
    print("result = ", result, sep='\n', end='\n\n')
    ...


def debug_pretrained_model():
    """
    官方训练好的模型。hf chatglm2-6b 代码
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(tokenizer)

    # 模型太大，无法调试
    exit()
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])

    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)

if __name__ == "__main__":
    # debug_tokenizer()
    # debug_model_config()
    debug_model()
    ...
