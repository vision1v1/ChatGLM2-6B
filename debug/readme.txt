# 模型结构
ChatGLMForConditionalGeneration ---> 特定任务架构
    1 ChatGLMModel ---> 模型的整体架构
        1 Embedding
        2 RotaryEmbedding
        3 GLMTransformer ---> encoder
            1 GLMBlock xN
                1 RMSNorm
                2 SelfAttention
                    1 Linear
                    2 CoreAttention  ---> 具体计算Attention的
                    3 Linear
                3 RMSNorm
                4 MLP
            2 RMSNorm
        4 Linear


# RotaryEmbedding
参考：https://kexue.fm/archives/8265



