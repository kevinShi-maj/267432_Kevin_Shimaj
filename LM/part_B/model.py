# Part 1.B — Manual LoRA (Low-Rank Adaptation) on pre-trained GPT2.
# Reference: Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
#            (arXiv:2106.09685). PEFT and similar libraries are NOT used — every
#            adapter below is hand-written.
#
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class LoRALinear(nn.Module):
    """One low-rank adapter for a single projection (Q, K or V).

    Implements the LoRA reparametrisation from Hu et al. 2021, §4.1:

        h = W0 x + ΔW x,   ΔW = B A,   ΔW x scaled by (alpha / rank)

    A : d_model -> rank   (down-projection), init ~ N(0,1)  (paper: random Gaussian)
    B : rank -> d_model   (up-projection),   init = 0

    Because B = 0 at init, ΔW = B A = 0, so the adapted model starts EXACTLY at the
    pre-trained weights (paper §4.1: "ΔW = 0 at the beginning of training"). The
    factor alpha/rank makes tuning alpha roughly equivalent to tuning the learning
    rate and lets us change rank without re-tuning everything (paper §4.1).
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
        # A random Gaussian, B zero -> ΔW = 0 at step 0 (Hu et al. §4.1).
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class CustomGPT2Attention(GPT2Attention):
    """GPT2 self-attention with LoRA on the Q, K, V projections.

    GPT2 packs Q, K, V into a single Conv1D `c_attn` mapping d_model -> 3*d_model.
    We keep `c_attn` frozen and add three independent LoRA deltas (one per Q/K/V),
    all computed from the same `hidden_states`, immediately after the split — this
    is the transformer-specific placement recommended in the paper (§4.2: "we limit
    our study to only adapting the attention weights").
    """

    def __init__(self, config, rank, alpha):
        super().__init__(config)
        d_model = config.hidden_size
        self.lora_q = LoRALinear(d_model, rank, alpha)
        self.lora_k = LoRALinear(d_model, rank, alpha)
        self.lora_v = LoRALinear(d_model, rank, alpha)

    # forward copied from transformers v4.38.0; LoRA deltas injected in the
    # self-attention branch only (cross-attention branch left untouched).
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
            # ---- LoRA injection: h = W0 x + (alpha/rank) B A x, for Q, K, V ----
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(hidden_states)
            value = value + self.lora_v(hidden_states)
            # -------------------------------------------------------------------

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
    """Pre-trained GPT2-small with every attention block replaced by a LoRA one.

    Build with `GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=r, alpha=a)`.
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        self.lora_rank = rank
        self.lora_alpha = alpha
        for block in self.transformer.h:
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # copy the pre-trained attention weights (c_attn/c_proj/bias); strict=False
            # because the LoRA layers are new and absent from the old state_dict.
            new_attn.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = new_attn

    def _init_weights(self, module):
        # from_pretrained() re-initialises every "missing" parameter (all lora_*)
        # via _init_weights AFTER __init__. GPT2's default _init_weights treats
        # lora_B as a plain nn.Linear and would fill it with N(0, 0.02), destroying
        # the B=0 property that guarantees ΔW=0 at start. We override to re-assert
        # the paper's init on LoRALinear. (Pitfall: "B not zero-initialized".)
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
