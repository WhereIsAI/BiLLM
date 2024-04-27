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

    from billm import OpenELMForCausalLM, OpenELMConfig

    model = OpenELMForCausalLM(OpenELMConfig(vocab_size=128,
                                             head_dim=32,
                                             num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biopenelm_seq_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import OpenELMForSequenceClassification, OpenELMConfig

    model = OpenELMForSequenceClassification(OpenELMConfig(vocab_size=128,
                                                           head_dim=32,
                                                           num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0


def test_biopenelm_token_clf():
    import os
    os.environ['BiLLM_START_INDEX'] = '1'

    from billm import OpenELMForTokenClassification, OpenELMConfig

    model = OpenELMForTokenClassification(OpenELMConfig(vocab_size=128,
                                                        head_dim=32,
                                                        num_transformer_layers=2))
    assert model is not None
    assert len(model.model.bidirectionas) > 0
