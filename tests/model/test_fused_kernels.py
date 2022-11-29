import math

import torch

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from transformers import BertTokenizer, GPT2Tokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import transformers

transformers.logging.set_verbosity(
    transformers.logging.FATAL,
)


def test_load_fused_kernels():
    try:
        import scaled_masked_softmax_cuda
        import scaled_upper_triang_masked_softmax_cuda
        import torch

        print("[Success] load_fused_kernels")
    except ImportError as e:
        print("[Fail] load_fused_kernels")
        raise e


def test_fused_softmax():
    from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
    from megatron.model.gpt2_model import (
        gpt2_attention_mask_func as attention_mask_func,
    )

    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi hi hi hi hi hi hi"  # 32
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    embedding_output = bert.embeddings(
        input_ids=tokens["input_ids"].cuda(),
        position_ids=None,
        token_type_ids=tokens["token_type_ids"].cuda(),
        inputs_embeds=None,
        past_key_values_length=0,
    )

    # (bsz, 1, 1, seq_len)
    mask = bert.get_extended_attention_mask(
        attention_mask=tokens["attention_mask"].cuda(),
        input_shape=tokens["input_ids"].shape,
        device=bert.device,
    )
    # (bsz, 1, seq_len, seq_len)
    mask = mask.repeat(1, 1, mask.size()[-1], 1)

    attention = bert.encoder.layer[0].attention.self
    key_layer = attention.transpose_for_scores(attention.key(embedding_output))
    query_layer = attention.transpose_for_scores(attention.query(embedding_output))

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores /= math.sqrt(key_layer.size()[-1])

    fused_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            fusion_type=SoftmaxFusionTypes.general,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attention_scores,
        (mask != 0),
    )

    torch_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            fusion_type=SoftmaxFusionTypes.none,
            scale=None,
            softmax_in_fp32=False,
        )
        .cuda()
        .half()
    )

    torch_softmax_output = torch_softmax(
        attention_scores,
        (mask != 0),
    )

    test_result = (fused_softmax_output - torch_softmax_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_softmax"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_softmax"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )


def test_fused_upper_triangle_mask_softmax():
    from megatron.model.gpt2_model import (
        gpt2_attention_mask_func as attention_mask_func,
    )
    from megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes

    gpt = GPT2Model.from_pretrained("gpt2").cuda().half()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi"  # 24
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    attention_mask = tokens["attention_mask"].cuda()
    attention_mask = attention_mask.view(attention_mask.size(0), -1)
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -10000.0
    attention_mask = attention_mask.repeat(1, 1, attention_mask.size()[-1], 1)
    attn = gpt.h[0]

    hidden_states = gpt.wte(tokens["input_ids"].cuda())
    q, k, v = attn.attn.c_attn(hidden_states).split(768, dim=-1)
    q = attn.attn._split_heads(q, attn.attn.num_heads, attn.attn.head_dim)
    k = attn.attn._split_heads(k, attn.attn.num_heads, attn.attn.head_dim)
    attn_weights = torch.matmul(q, k.transpose(-1, -2))

    sq, sk = q.size(-2), k.size(-2)
    causal_mask = attn.attn.bias[:, :, sk - sq : sk, :sk].bool()
    total_mask = ~(causal_mask & (attention_mask == 0))
    """
    tensor([[[[False,  True,  True,  ...,  True,  True,  True],
              [False, False,  True,  ...,  True,  True,  True],
              [False, False, False,  ...,  True,  True,  True],
              ...,
              [False, False, False,  ..., False,  True,  True],
              [False, False, False,  ..., False, False,  True],
              [False, False, False,  ..., False, False, False]]]
    """

    fused_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            fusion_type=SoftmaxFusionTypes.upper_triang,
            scale=None,
            softmax_in_fp32=False,
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attn_weights,
        total_mask,
    )

    torch_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            fusion_type=SoftmaxFusionTypes.none,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
        )
        .cuda()
        .half()
    )

    torch_softmax_output = torch_softmax(
        attn_weights,
        total_mask,
    )

    test_result = (fused_softmax_output - torch_softmax_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
