from typing import Optional, Tuple, Union
import torch.nn as nn
from transformers import WhisperModel
from transformers import WhisperConfig


class CustomWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

    def forward(
        self,
        input_features,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        head_mask,
        decoder_head_mask,
        cross_attn_head_mask,
        encoder_outputs,
        past_key_values,
        decoder_inputs_embeds,
        decoder_position_ids,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        cache_position,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_features = self._mask_input_features(
            input_features, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=torch.cat(
                (encoder_outputs[0], encoder_outputs[0]), dim=0
            ),
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        return decoder_outputs + encoder_outputs


class CustomWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomWhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions
        self.post_init()
