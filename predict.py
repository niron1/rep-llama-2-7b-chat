# clean version of llama model, system prompt is in the hands of developer
# temperature has effect in this version
# streaming support

from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import re
from datetime import datetime


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "daryl149/Llama-2-7b-chat-hf",
            use_cache="cache"
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = AutoModelForCausalLM.from_pretrained(
            "daryl149/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_cache="cache"
        ).to(self.device)

        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.eos_token_id

        self.model = model


    def predict(
        self,
        max_new_tokens: int = Input(description="max_new_tokens", default=1000 ),
        temperature: float = Input(description="temperature", default=0.9 ),
        repetition_penalty: float = Input(description="repetition_penalty", default=1.1),
        prompt: str = Input(description="prompt", default="Can ducks fly?"),
    ) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        streamer = TextStreamer(self.tokenizer)
        seed = int(datetime.now().timestamp())
        print("seed", seed)
        torch.manual_seed(seed)
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens,temperature=temperature, repetition_penalty=repetition_penalty, streamer=streamer, do_sample=True)
        output = self.tokenizer.decode(outputs[0])
        output = re.sub(r'<s>|</s>', '', output)
        return output
