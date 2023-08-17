# clean version of llama model, system prompt is in the hands of developer
# temperature has effect in this version
# streaming support

from cog import BasePredictor, Input, ConcatenateIterator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from stream_search import stream_search
from datetime import datetime
from threading import Thread

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

        # model.config.pad_token_id = model.config.eos_token_id
        # model.generation_config.pad_token_id = model.config.eos_token_id

        self.model = model


    def predict(
        self,
        prompt: str = Input(description="prompt", default="Can ducks fly?"),
        max_new_tokens: int = Input(description="max_new_tokens", default=1000),
        temperature: float = Input(description="temperature", default=0.9),
        seed: int = Input(description="random number seed, -1=generate", default=-1),
        repetition_penalty: float = Input(description="repetition_penalty", default=1.1),
    ) -> ConcatenateIterator[str]:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer)
        if seed == -1:
            seed = int(datetime.now().timestamp())
        print("seed", seed)

        print("model config eos_token_id", self.model.config.eos_token_id)
        print("model config pad_token_id", self.model.config.pad_token_id)
        print("model generation config pad_token_id", self.model.generation_config.pad_token_id)

        torch.manual_seed(seed)
        generation_kwargs = dict(inputs=inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                 repetition_penalty=repetition_penalty, streamer=streamer, do_sample=True)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in stream_search(['<s> ','</s>'],streamer):
            yield new_text

        # output = self.tokenizer.decode(outputs[0])
        # no_padding = re.sub(r'<s> |</s>', '', output)
        # unique_output = no_padding[len(prompt):]
        # return unique_output.strip()
