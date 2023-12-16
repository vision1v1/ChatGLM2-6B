import os
data_dir = os.getenv("my_data_dir")
model_path = os.path.normpath(os.path.join(data_dir, 'pretrained', 'THUDM/chatglm2-6b'))
print(model_path)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(tokenizer)

exit()
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()


model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])

print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
