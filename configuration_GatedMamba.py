"""GatedMamba config for Hugging Face.

"""

from typing import Optional, Union

from transformers import PretrainedConfig


class GatedMambaConfig(PretrainedConfig):
    model_type = "GatedMamba"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            d_intermediate: int = 0,                   
            n_layer: int = 64,
            attn_layer_idx: Optional[list] = None,      
            MSC_layer_idx: Optional[list] = None,      
            dilation_base: int = 4,     
            dropout: float = 0.0,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            attn_cfg: Optional[dict] = None,        
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,

            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            complement_map: Optional[dict] = None,  
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.attn_layer_idx = attn_layer_idx
        self.MSC_layer_idx = MSC_layer_idx
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.attn_cfg = attn_cfg
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.complement_map = complement_map
