# -*- coding: utf-8 -*-


def test_openelm_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '-1'

    from billm import OpenELMModel, OpenELMConfig

    model = OpenELMModel(OpenELMConfig(vocab_size=128,
                                       head_dim=32,
                                       num_transformer_layers=2))
    assert model is not None


def test_biopenelm_model():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import OpenELMModel, OpenELMConfig

    model = OpenELMModel(OpenELMConfig(vocab_size=128,
                                       head_dim=32,
                                       num_transformer_layers=2))
    assert model is not None



def test_biopenelm_lm():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForCausalLM, OpenELMConfig

    model = Qwen2ForCausalLM(OpenELMConfig(vocab_size=128,
                                             head_dim=32,
                                             intermediate_size=64,
                                             num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biopenelm_seq_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForSequenceClassification, OpenELMConfig

    model = Qwen2ForSequenceClassification(OpenELMConfig(vocab_size=128,
                                                           head_dim=32,
                                                           num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biopenelm_token_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import Qwen2ForTokenClassification, OpenELMConfig

    model = Qwen2ForTokenClassification(OpenELMConfig(vocab_size=128,
                                                        head_dim=32,
                                                        num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0
