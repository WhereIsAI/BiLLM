# -*- coding: utf-8 -*-


def test_llama_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '-1'

    from billm import LlamaModel, LlamaConfig

    model = LlamaModel(LlamaConfig(vocab_size=128,
                                   hidden_size=32,
                                   intermediate_size=64,
                                   num_hidden_layers=2,
                                   num_attention_heads=2))
    assert model is not None


def test_billama_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import LlamaModel, LlamaConfig

    model = LlamaModel(LlamaConfig(vocab_size=128,
                                   hidden_size=32,
                                   intermediate_size=64,
                                   num_hidden_layers=2,
                                   num_attention_heads=2))
    assert model is not None



def test_billama_lm():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import LlamaForCausalLM, LlamaConfig

    model = LlamaForCausalLM(LlamaConfig(vocab_size=128,
                                         hidden_size=32,
                                         intermediate_size=64,
                                         num_hidden_layers=2,
                                         num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_billama_seq_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import LlamaForSequenceClassification, LlamaConfig

    model = LlamaForSequenceClassification(LlamaConfig(vocab_size=128,
                                                       hidden_size=32,
                                                       intermediate_size=64,
                                                       num_hidden_layers=2,
                                                       num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_billama_token_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import LlamaForTokenClassification, LlamaConfig

    model = LlamaForTokenClassification(LlamaConfig(vocab_size=128,
                                                    hidden_size=32,
                                                    intermediate_size=64,
                                                    num_hidden_layers=2,
                                                    num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0
