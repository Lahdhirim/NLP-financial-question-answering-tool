class DataSchema:
    QUESTION = "question"
    CONTEXT = "context"
    ANSWER = "answer"

class ModelSchema:
    INPUT_IDS = "input_ids"
    ENCODER_MASK = "encoder_attention_mask"
    LABELS = "labels"
    DECODER_MASK = "decoder_attention_mask"

class CheckpointSchema:
    EPOCH = "epoch"
    VAL_LOSS = "val_loss"
    STATE_DICT = "state_dict"
    MODEL_NAME = "model_name"
    TOKENIZER_PRETRAINED_MODEL = "tokenizer_pretrained_model"
    MAX_INPUT_LENGTH = "max_input_length"
    MAX_ANSWER_LENGTH = "max_answer_length"

class MetricSchema:
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    BLEU1 = "bleu1"
    BLEU2 = "bleu2"
