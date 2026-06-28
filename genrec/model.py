import torch.nn as nn

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class AbstractModel(nn.Module):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(AbstractModel, self).__init__()

        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        
    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def calculate_loss(self, batch):
        raise NotImplementedError('calculate_loss method must be implemented.')

    def generate(self, batch, n_return_sequences=1):
        raise NotImplementedError('predict method must be implemented.')

    def on_train_epoch_end(self, epoch: int) -> dict[str, float]:
        """Hook called after each training epoch, before validation.

        Override to run model-specific epoch-end logic (e.g. temperature
        annealing) and return scalar metrics to log. Keys are W&B metric paths;
        the trainer logs them to W&B and the console.

        Args:
            epoch: Zero-based index of the epoch that just finished.

        Returns:
            Mapping of metric name to scalar value, e.g.
            ``{"Quantizer/gumbel_tau": 0.5}``. Empty by default.
        """
        return {}

    def prepare_before_test(self):
        """Hook called before testing.

        Override to run model-specific setup before testing.
        """
        pass