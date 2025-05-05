import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
logger = logging.getLogger(__name__)


class LLaVA_seaLLMs(nn.Module):
    def __init__(
        self,
        model_name="SeaLLMs/SeaLLMs-v3-7B",
        device_map="auto",
        delay_load=False,
        is_loaded=False,
        requires_grad=False
    ):
        super(LLaVA_seaLLMs, self).__init__()
        self.is_loaded = is_loaded
        self.model_name = model_name
        self.delay_load = delay_load
        self.requires_grad = requires_grad
        self.device_map = self._decide_device_map(device_map)
        try:
            self.compute_dtype = torch.bfloat16
            _ = torch.zeros(1, dtype=torch.bfloat16)
            logger.info("Usng bfloat16 for LLM")
        except RuntimeError:
            self.compute_dtype = torch.float32
            logger.warning(
                "bfloat16 not supported, falling back to float16. This might be less stable.")

        self.model = None
        self.tokenizer = None
        if not self.delay_load:
            self.load_model()

    def _decide_device_map(self, device_map):
        """Decide device map based on available hardware"""
        if device_map is not None:
            return device_map  # user override

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                logger.info(
                    f"Found {gpu_count} GPUs. Using 'balanced' device map.")
                return "balanced"
            else:
                logger.info("Found 1 GPU. Using 'auto' device map.")
                return "auto"
        else:
            logger.info("No CUDA GPUs found. Using 'cpu' device map.")
            return "cpu"

    def load_model(self):
        """Load the model and tokenizer."""
        if self.is_loaded:
            logger.info(f"Model {self.model_name} already loaded. Skipping.")
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.compute_dtype,
            output_hidden_states=True
        )
        self.model.eval()
        self.model.requires_grad_(self.requires_grad)
        # self.model.to(self._device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.embed_dim = self.model.config.hidden_size
        self.max_seq_length = self.model.config.max_position_embeddings
        self.vocab_size = self.model.config.vocab_size
        self.is_loaded = True

    def tokenize(self, text, return_tensors="pt"):
        """
        Tokenize the text and prepare it for the model
        """
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        device = next(self.model.parameters()).device

        return {k: v.to(device) for k, v in tokens.items()}

    def embeddings(self, input_ids):
        """
        Forward pass for obtaining the embeddings of text.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for padding.

        Returns:
            torch.Tensor: The embeddings of the text.
        """
        outputs = self.model(**input_ids)

        # Return the embeddings of the last hidden state
        return outputs.hidden_states[-1].to(self.compute_dtype)

    def forward(self, text):
        """The forward method to pass the text and get the embeddings."""
        inputs = self.tokenize(text)
        text_embeddings = self.embeddings(inputs)
        return text_embeddings

    @property
    def embed_dims(self):
        if not self.is_loaded:
            self.load_model()
        return self.embed_dim

    @property
    def dtype(self):
        return self.compute_dtype
