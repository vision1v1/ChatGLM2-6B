from tokenization_chatglm import ChatGLMTokenizer, SPTokenizer
import os

data_dir = os.getenv("my_data_dir")
model_path = os.path.normpath(os.path.join(data_dir, 'pretrained', 'THUDM', 'chatglm2-6b'))

def debug_tokenizer():
    """
    调试分词，使用的 SentencePieceProcessor
    TODO 如何训练 tokenizer.model
    """

    txt = '同事们好！'

    # tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    tokenizer = ChatGLMTokenizer(vocab_file=os.path.join(model_path, 'tokenizer.model'))
    tokens = tokenizer.tokenize(txt)  # 分词
    print("tokens = ", tokens, sep='\n', end='\n\n')

    token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
    print("token_ids = ", token_ids, sep='\n', end='\n\n')


    sp_tokenizer = SPTokenizer(model_path=os.path.join(model_path, 'tokenizer.model'))
    tokens = sp_tokenizer.tokenize(txt)
    print("tokens = ", tokens, sep='\n', end='\n\n')

    
    token_ids = [sp_tokenizer.convert_token_to_id(token=token) for token in tokens]
    print("token_ids = ", token_ids, sep='\n', end='\n\n')


def debug_sptokenizer():
    sp_tokenizer = SPTokenizer(model_path=os.path.join(model_path, 'tokenizer.model'))

    txt = '同事们好！'
    tokens = sp_tokenizer.tokenize(txt)
    print("tokens = ", tokens, sep='\n', end='\n\n')


if __name__ == "__main__":
    debug_tokenizer()
    # debug_sptokenizer()