from typing import Optional

import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import reduce


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask = mask.to(dtype=torch.bool, device=logits.device)
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()

        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(self.mask,
                              p_log_p,
                              torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)

    def argmax(self) -> torch.Tensor:
        return self.logits.sort(dim=1, descending=True)[1][:, 0]  # 按照概率值倒排,选择最大概率位置的索引


if __name__ == "__main__":
    logits_or_qvalues = torch.randn((2, 3))
    print(logits_or_qvalues)

    mask = torch.zeros((2, 3), dtype=torch.bool)
    mask[0][2] = True
    mask[1][0] = True
    mask[1][1] = True
    print(mask)

    head = CategoricalMasked(logits=logits_or_qvalues)
    print(head.probs)

    head_masked = CategoricalMasked(logits=logits_or_qvalues, mask=mask)
    print(head_masked.probs)

    print(head.entropy())
    print(head_masked.entropy())
