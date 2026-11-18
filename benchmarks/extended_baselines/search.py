"""Beam search over distributions captured from the authors' greedy path."""
import torch
from native_runtime import readonly_observation
from native_generation import search, tokenize_response


class Ready(Exception):
    pass


class Distribution:
    def __init__(self, runtime, batch):
        self.runtime, self.model = runtime, runtime.model
        model = self.model
        self.decoder = model.decoder.forward
        self.generator = model.generator.forward
        self.rel_decoder = model.rel_decoder.forward if runtime.family == "iamm" else None
        def decoder(*args, **kwargs):
            self.decoder_args, self.decoder_kwargs = args, kwargs
            return self.decoder(*args, **kwargs)
        def rel_decoder(*args, **kwargs):
            self.rel_args, self.rel_kwargs = args, kwargs
            return self.rel_decoder(*args, **kwargs)
        def generator(*args, **kwargs):
            self.generator_args, self.generator_kwargs = args, kwargs
            self.first = self.log_probs(self.generator(*args, **kwargs))[0, -1].detach()
            raise Ready()
        model.decoder.forward, model.generator.forward = decoder, generator
        if self.rel_decoder:
            model.rel_decoder.forward = rel_decoder
        try:
            runtime.greedy(batch, 1)
        except Ready:
            pass
        finally:
            model.decoder.forward, model.generator.forward = self.decoder, self.generator
            if self.rel_decoder:
                model.rel_decoder.forward = self.rel_decoder

    def log_probs(self, value):
        return value.log() if self.runtime.family == "escm" else value

    def step(self, tokens):
        if not tokens:
            return self.first
        ids = torch.tensor([[3] + list(tokens)], device=self.runtime.device)
        embedded = self.model.embedding(ids)
        args = list(self.decoder_args)
        args[0] = embedded
        args[2] = (args[2][0], ids.eq(1).unsqueeze(1))
        output, attention = self.decoder(*args, **self.decoder_kwargs)
        if self.rel_decoder:
            args = list(self.rel_args)
            args[0] = embedded
            args[2] = (args[2][0], ids.eq(1).unsqueeze(1))
            rel, _ = self.rel_decoder(*args, **self.rel_kwargs)
            gate = torch.sigmoid(self.model.rs_logit_linear(torch.cat((output, rel), dim=-1)))
            output = gate * output + (1 - gate) * rel
        args = list(self.generator_args)
        args[0], args[1] = output, attention
        return self.log_probs(self.generator(*args, **self.generator_kwargs))[0, -1]


def beam(runtime, batch, max_steps):
    with readonly_observation(runtime.model):
        ids = search(Distribution(runtime, batch), max_steps, 5)
        content = ids[:ids.index(2)] if 2 in ids else ids
        text = " ".join(runtime.vocab.index2word[i] for i in content)
        return dict(beam_text=text, beam_tokens=tokenize_response(text))
