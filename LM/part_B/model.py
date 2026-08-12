from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class LoRALinear(nn.Module):
    """Low-rank adapter for one projection: h = W0 x + (alpha / rank) * B(A(x)).

    The rank bottleneck encodes the paper's assumption that the weight update needed
    to adapt the model has a low intrinsic rank. Scaling by alpha / rank keeps the
    size of the update roughly constant when rank changes, so the two can be tuned
    independently.
    """

    def __init__(self, d_model: int, rank: int, alpha: int):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_model, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Zeroing B makes the whole update B(A(x)) vanish at step 0, so fine-tuning
        # starts exactly at the pre-trained weights instead of perturbing them.
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class CustomGPT2Attention(GPT2Attention):
    """GPT2 self-attention with a separate LoRA adapter on each of Q, K and V.

    GPT2 computes the three projections with a single Conv1D (`c_attn`, d_model to
    3 * d_model), so the adapters cannot wrap them individually. They are instead
    applied to the same input and added to the three slices after the split, which
    is equivalent to adapting Wq, Wk and Wv separately.
    """

    def __init__(self, config, rank, alpha):
        super().__init__(config)
        d_model = config.hidden_size
        self.lora_q = LoRALinear(d_model, rank, alpha)
        self.lora_k = LoRALinear(d_model, rank, alpha)
        self.lora_v = LoRALinear(d_model, rank, alpha)

    # Body copied verbatim from transformers v4.38.0 so that the attention internals
    # stay in sync with the pinned version; only the three additions below are ours.
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(hidden_states)
            value = value + self.lora_v(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2_LoRA(GPT2LMHeadModel):
    """Pre-trained GPT2 whose attention blocks are replaced by LoRA-adapted ones.

    Build with `GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=r, alpha=a)`.
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        self.lora_rank = rank
        self.lora_alpha = alpha
        for block in self.transformer.h:
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # strict=False: the adapters are new, so they have no entry in the old
            # attention state_dict and would otherwise be reported as missing keys.
            new_attn.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = new_attn

    def _init_weights(self, module):
        # from_pretrained() re-initialises whatever the checkpoint does not provide,
        # which includes every adapter. Left to the default GPT2 rule, lora_B would be
        # filled from a normal distribution and the update would no longer be zero at
        # step 0, so the adapter init has to be re-asserted here.
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()
