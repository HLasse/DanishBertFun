# DanishBertFun

An attempt to finetune [Danish Bert](https://github.com/botxo/danish_bert) for NER

## Convert tf checkpoint to pytorch

python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path model.ckpt --bert_config_file config.json --pytorch_dump_path pytorch_model.bin

