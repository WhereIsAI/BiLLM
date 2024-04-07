# -*- coding: utf-8 -*-


def test_qwen2_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '-1'

    from billm import Qwen2Model, Qwen2Config

    model = Qwen2Model(Qwen2Config(vocab_size=128,
                                       hidden_size=32,
                                       intermediate_size=64,
                                       num_hidden_layers=2,
                                       num_attention_heads=2))
    assert model is not None


def test_biqwen2_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2Model, Qwen2Config

    model = Qwen2Model(Qwen2Config(vocab_size=128,
                                       hidden_size=32,
                                       intermediate_size=64,
                                       num_hidden_layers=2,
                                       num_attention_heads=2))
    assert model is not None



def test_biqwen2_lm():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForCausalLM, Qwen2Config

    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=128,
                                             hidden_size=32,
                                             intermediate_size=64,
                                             num_hidden_layers=2,
                                             num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biqwen2_seq_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForSequenceClassification, Qwen2Config

    model = Qwen2ForSequenceClassification(Qwen2Config(vocab_size=128,
                                                           hidden_size=32,
                                                           intermediate_size=64,
                                                           num_hidden_layers=2,
                                                           num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biqwen2_token_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForTokenClassification, Qwen2Config

    model = Qwen2ForTokenClassification(Qwen2Config(vocab_size=128,
                                                        hidden_size=32,
                                                        intermediate_size=64,
                                                        num_hidden_layers=2,
                                                        num_attention_heads=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0
