import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from configuration_chatglm import ChatGLMConfig
from modeling_chatglm import ChatGLMModel, ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer, SPTokenizer
import torch

torch.set_printoptions(linewidth=500)  # 方便阅读

data_dir = os.getenv("my_data_dir")
model_path = os.path.normpath(os.path.join(data_dir, 'pretrained', 'THUDM', 'chatglm2-6b'))
print(model_path)


def debug_tokenizer():
    """
    调试分词，使用的 SentencePieceProcessor
    TODO 如何训练 tokenizer.model
    """
    # tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    tokenizer = ChatGLMTokenizer(vocab_file=os.path.join(model_path, 'tokenizer.model'))

    txt = '同志们好！'
    tokens = tokenizer.tokenize(txt)  # 分词
    print("tokens = ", tokens, sep='\n', end='\n\n')

    sp_tokenizer = SPTokenizer(model_path=os.path.join(model_path, 'tokenizer.model'))
    tokens = sp_tokenizer.tokenize(txt)
    print("tokens = ", tokens, sep='\n', end='\n\n')

    # token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
    # print("token_ids = ", token_ids, sep='\n', end='\n\n')

    # input_ids = tokenizer.encode(text=txt)
    # print("input_ids = ", input_ids, sep='\n', end='\n\n')

    # print(tokenizer.convert_ids_to_tokens(input_ids))
    ...


def debug_model_config():
    """
    调试chatglm2-6b的模型配置
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("AutoConfig.from_pretrained = ", config, sep='\n', end='\n\n')

    config = ChatGLMConfig.from_pretrained(model_path)
    print("ChatGLMConfig.from_pretrained = ", config, sep='\n', end='\n\n')

    config = ChatGLMConfig()
    print("ChatGLMConfig() = ", config, sep='\n', end='\n\n')


def debug_model_forward():
    """
    模型在 modeling_chatglm.py 中。 
    """
    config = ChatGLMConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float32  # 在cpu上调试，所以这里改为float32

    # 为了能跑起来，降低模型复杂度
    config.ffn_hidden_size = 1024
    config.num_attention_heads = 2
    config.num_layers = 3

    model = ChatGLMModel(config=config)
    tokenizer = ChatGLMTokenizer(vocab_file=os.path.join(model_path, 'tokenizer.model'))

    txt = '同志们好！'
    input_ids = tokenizer.encode(txt, return_tensors='pt')

    result = model.forward(input_ids)

    for key, value in result.items():
        if torch.is_tensor(value):
            print(key, value, sep='\n', end='\n\n')
        if isinstance(value, tuple):
            for idx, (past_key, past_value) in enumerate(value):
                print(f"{key} {idx}", "past_key = ", past_key, "past_value = ", past_value, sep='\n', end='\n\n')


def debug_gen():
    """
    调试条件生成逻辑。为了调试方便，不加载训练好的模型
    """
    # 模型配置
    config = ChatGLMConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float32  # 在cpu上调试，所以这里改为float32
    # 为了能跑起来，降低模型复杂度
    config.ffn_hidden_size = 1024
    config.num_attention_heads = 2
    config.num_layers = 3

    # 生成式模型
    model = ChatGLMForConditionalGeneration(config)

    # 模拟数据
    query = "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"
    answer = "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。"

    # 模型输入构建
    tokenizer: ChatGLMTokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    prompt = tokenizer.build_prompt(query, history="")

    max_source_length = 64
    max_target_length = 128
    max_seq_length = max_source_length + max_target_length + 1
    # 只做截断不做填充，因为下面要做拼接。
    a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_source_length) # encode 返回的就是 input_ids
    b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, max_length=max_target_length)

    context_length = len(a_ids)  # 上下文长度
    input_ids = a_ids + b_ids + [tokenizer.eos_token_id]  # 提示词 + 答案 + 结束词
    labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]  # 填充 + 答案 + 结束词

    pad_len = max_seq_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len  # input_ids 对齐到最大长度
    labels = labels + [tokenizer.pad_token_id] * pad_len  # labels 对齐到最大长度
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]  # labels中pad的词忽略掉，不计算loss

    model_inputs = {
        "input_ids": [input_ids],
        "labels": [labels],
    }

    model_inputs = tokenizer.pad(model_inputs, padding=False, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
    
    
    # forward
    output:CausalLMOutputWithPast = model.forward(**model_inputs)
    print("loss", output.loss, "logits", output.logits, sep='\n', end='\n\n')


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
    # debug_model_forward()
    debug_gen()
    ...
