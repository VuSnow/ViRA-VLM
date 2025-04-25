import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaVA_seaLLMs(nn.Module):
    def __init__(
        self,
        model_name="SeaLLM/SeaLLM-7B",
        device=None,
        delay_load=False,
        is_loaded=False,
    ):
        super(LLaVA_seaLLMs, self).__init__()
        self.is_loaded = is_loaded
        self.model_name = model_name
        self.delay_load = delay_load
        self._device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        if not self.delay_load:
            self.load_model()

    def load_model(self):
        """Load the model and tokenizer."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.embed_dim = self.model.config.hidden_size
        self.max_seq_length = self.model.config.max_position_embeddings
        self.vocab_size = self.model.config.vocab_size
        self.is_loaded = True
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self._device)

    def encode(self, text, return_tensors="pt"):
        """Tokenize the text and prepare it for the model."""
        if isinstance(text, str):
            text = [text]
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

    def embeddings(self, input_ids, attention_mask=None):
        """
        Forward pass for obtaining the embeddings of text.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for padding.

        Returns:
            torch.Tensor: The embeddings of the text.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        # Return the embeddings of the last hidden state
        return outputs.last_hidden_state

    def forward(self, text):
        """The forward method to pass the text and get the embeddings."""
        encoding = self.encode(text)
        input_ids = encoding['input_ids'].to(self._device)
        attention_mask = encoding['attention_mask'].to(self._device)

        # Get text embeddings (using the last hidden state from the transformer model)
        text_embeddings = self.embeddings(input_ids, attention_mask)

        return text_embeddings
