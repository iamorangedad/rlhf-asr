import torch
from transformers import Trainer


class ORPOTrainer(Trainer):
    def __init__(self, alpha, pad, disable_prompt_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = pad
        self.pad_token_id = pad
        self.alpha = alpha
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self.disable_prompt_loss = disable_prompt_loss
        print("Pad Token ID: ", self.pad)

    def compute_custom_loss(self, logits, labels):
        logits = logits.contiguous()
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(
                dim=-1
            )
        return loss

    def compute_logps(self, chosen_inputs, logits, pad_token_id):
        if False:
            mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
            per_token_logps = torch.gather(
                logits[:, :-1, :].log_softmax(-1),
                dim=2,
                index=(mask * chosen_inputs[:, 1:]).unsqueeze(2),
            ).squeeze(2)
            return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(
                dim=1
            ).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
        labels = chosen_inputs[:, 1:].clone()
        mask = labels != pad_token_id
        labels = torch.where(labels == pad_token_id, 0, labels)
        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(
            dtype=torch.float64
        ) / mask.sum(dim=1).to(dtype=torch.float64)

    def compute_loss(self, model, inputs, return_outputs=False):
        # model outputs
        neg_labels = inputs["negative_input_ids"].clone()
        pos_labels = inputs["positive_input_ids"].clone()
        outputs_neg = model(
            **{
                "input_features": inputs["input_features"],
                "labels": neg_labels,
            },
            output_hidden_states=True
        )
        outputs_pos = model(
            **{
                "input_features": inputs["input_features"],
                "labels": pos_labels,
            },
            output_hidden_states=True
        )
        # Calculate NLL loss
        pos_loss = outputs_pos.loss
        # Calculate Log Probability
        pos_prob = self.compute_logps(
            chosen_inputs=inputs["positive_input_ids"],
            logits=outputs_pos.logits,
            pad_token_id=-100,
        )
        neg_prob = self.compute_logps(
            chosen_inputs=inputs["negative_input_ids"],
            logits=outputs_neg.logits,
            pad_token_id=-100,
        )
        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (
            torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob))
        )
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        # Calculate the Final Loss
        loss = torch.mean(pos_loss - self.alpha * ratio).to(dtype=torch.bfloat16)
        return (loss, outputs_pos) if return_outputs else loss
