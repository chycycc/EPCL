"""Common search over each model's own reference-free decoder distribution.

Context is captured only from decoder_greedy, never from teacher forcing.
No native model parameters, losses, gradients or saved state are changed.
"""
import unicodedata

from native_runtime import readonly_observation


def tokenize_response(text):
    from nltk.tokenize import TreebankWordTokenizer
    return TreebankWordTokenizer().tokenize(unicodedata.normalize("NFKC", text).strip())


class Ready(Exception):
    pass


class NativeDistribution:
    def __init__(self, runtime, batch):
        self.runtime = runtime
        self.model = runtime.model
        decoder = self.model.decoder
        generator = self.model.generator
        self.decoder_forward, self.generator_forward = decoder.forward, generator.forward
        def decoder_capture(*args, **kwargs):
            self.decoder_args, self.decoder_kwargs = list(args), dict(kwargs)
            return self.decoder_forward(*args, **kwargs)
        def generator_capture(*args, **kwargs):
            self.generator_args, self.generator_kwargs = list(args), dict(kwargs)
            self.first_log_prob = self.generator_forward(*args, **kwargs)[:, -1].detach()
            raise Ready()
        decoder.forward, generator.forward = decoder_capture, generator_capture
        try:
            with readonly_observation(self.model):
                self.model.decoder_greedy(batch, max_dec_step=0)
        except Ready:
            pass
        finally:
            decoder.forward, generator.forward = self.decoder_forward, self.generator_forward
        if not hasattr(self, "first_log_prob"):
            raise ValueError("Native greedy path did not produce a decoder distribution")
        if self.first_log_prob.shape[-1] != runtime.vocabulary["n_words"]:
            raise ValueError("Unexpected generation probability support")

    def step(self, generated):
        import torch
        if not generated:
            return self.first_log_prob[0]
        ids = torch.tensor([[3] + list(generated)], dtype=torch.long, device=self.runtime.device)
        if self.runtime.name == "kemp":
            initial = self.decoder_kwargs["inputs"][:, :1]
            embedded = torch.cat((initial, self.model.embedding(ids[:, 1:])), dim=1)
        else:
            embedded = self.model.embedding(ids)
        args, kwargs = list(self.decoder_args), dict(self.decoder_kwargs)
        if "inputs" in kwargs:
            kwargs["inputs"] = embedded
        else:
            args[0] = embedded
        if "mask" in kwargs:
            kwargs["mask"] = (kwargs["mask"][0], ids.eq(1).unsqueeze(1))
        else:
            args[2] = (args[2][0], ids.eq(1).unsqueeze(1))
        output = self.decoder_forward(*args, **kwargs)
        generator_args = list(self.generator_args)
        generator_args[0] = output[0]
        generator_args[3 if self.runtime.name == "kemp" else 1] = output[1]
        return self.generator_forward(*generator_args, **self.generator_kwargs)[0, -1]


def search(distribution, max_tokens, beam_size):
    if max_tokens < 1 or beam_size < 1:
        raise ValueError("Positive search budget and beam width required")
    beams = [((), 0.0, False)]
    for _ in range(max_tokens):
        candidates = []
        for tokens, score, ended in beams:
            if ended:
                candidates.append((tokens, score, True))
                continue
            probabilities = distribution.step(tokens)
            if beam_size == 1:
                # Match native torch.max's first-index tie breaking exactly.
                value, index = probabilities.max(dim=-1)
                values, indices = value.reshape(1), index.reshape(1)
            else:
                values, indices = probabilities.topk(beam_size)
            for value, token in zip(values.tolist(), indices.tolist()):
                candidates.append((tokens + (token,), score + value, token == 2))
        beams = sorted(candidates, key=lambda value: (-value[1], value[0]))[:beam_size]
        if all(ended for _, _, ended in beams):
            break
    return list(beams[0][0])


def generate(runtime, batch, max_tokens=40, beam_size=5):
    import torch
    with readonly_observation(runtime.model):
        distribution = NativeDistribution(runtime, batch)
        ids = {"greedy": search(distribution, max_tokens, 1), "beam": search(distribution, max_tokens, beam_size)}
        result = {}
        for mode, values in ids.items():
            content = values[:values.index(2)] if 2 in values else values
            text = " ".join(runtime.vocabulary["index2word"][index] for index in content)
            result[mode + "_text"] = text
            result[mode + "_tokens"] = tokenize_response(text)
            result[mode + "_ids_including_eos"] = values
        # Independent native greedy implementation must agree at the same token budget.
        native_text = runtime.model.decoder_greedy(batch, max_dec_step=max_tokens - 1)[0].strip()
        native_text = native_text.replace("[USR]", "USR").replace("[SYS]", "SYS")
        if tokenize_response(native_text) != result["greedy_tokens"]:
            raise ValueError("Common greedy search does not match the native model")
        # A poisoned target must not change the generation context/distribution.
        poisoned = dict(batch)
        for name in ("target_batch", "target_ext_batch"):
            if name in poisoned:
                poisoned[name] = torch.full_like(poisoned[name], 3)
        reference_free = NativeDistribution(runtime, poisoned)
        if not torch.equal(distribution.first_log_prob, reference_free.first_log_prob):
            raise ValueError("Generation depends on teacher target tokens")
        result["native_greedy_equal"] = True
        result["poisoned_target_first_distribution_equal"] = True
        return result
