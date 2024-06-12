import os

import fire
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer

from sproto.dataset.outcome import OutcomeDiagnosesDataset, collate_batch
from sproto.model.multi_proto import MultiProtoModule
from sproto.utils.utils import ProjectorCallback, load_eval_buckets
from tqdm import tqdm
torch.set_float32_matmul_precision('high')

def check_val(val):
    for batch in tqdm(val):
        if not (set(batch['targets'].max(axis=1)).__len__() > 1):
            alarm = True
            return alarm
    return False


def run_training(
    train_file,
    val_file,
    test_file,
    label_column,
    batch_size=10,
    gpus=1,
    lr_prototypes=1e-3,
    lr_others=2e-2,
    lr_features=2e-6,
    num_warmup_steps=100,
    max_length=512,
    num_training_steps=5000,
    check_val_every_n_epoch=1,
    num_val_samples=None,
    pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    save_dir="output",
    projector_callback=False,
    model_type="PROTO",
    resume_from_checkpoint=None,
    use_sigmoid=False,
    seed=7,
    project_n_batches=1,
    use_attention=True,
    use_global_attention=False,
    dot_product=False,
    reduce_hidden_size=None,
    prototype_vector_path=None,
    attention_vector_path=None,
    all_labels_path=None,
    normalize=None,
    loss="BCE",
    final_layer=False,
    use_prototype_loss=False,
    eval_bucket_path=None,
    few_shot_experiment=False,
    metric_opt="auroc_macro",
    num_prototypes_per_class=1,
    pretrained_model_path=None,
    train_files=[],
    val_files=[],
    test_files=[],
    only_test=False,
    patience=25,

):
    pl.utilities.seed.seed_everything(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if few_shot_experiment:
        dataset = FilteredDiagnosesDataset  # noqa
    else:
        dataset = OutcomeDiagnosesDataset

    if not len(train_files):

        train_dataset = dataset(
            train_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )
    else:
        train_dataset = dataset(
            train_files, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )

    if not len(val_files):
        val_dataset = dataset(
            val_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )
    else:
        val_dataset = dataset(
            val_files, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )

    if not len(test_files):
        test_dataset = dataset(
            test_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )
    else:
        test_dataset = dataset(
            test_files, tokenizer, max_length=max_length, all_codes_path=all_labels_path,
            text_column="text", label_column=label_column
        )
    dataloader = {}
    for split, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
        dataloader[split] = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_batch,
            batch_size=batch_size,
            num_workers=5 if split == "train" else 5,
            pin_memory=True,
            shuffle=split == "train",
        )

    eval_buckets = load_eval_buckets(eval_bucket_path)

    if model_type == "BERT":
        model = BertModule(
            pretrained_model=pretrained_model,
            num_classes=dataset.get_num_classes(),
            lr_features=lr_features,
            lr_others=lr_others,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            save_dir=save_dir,
            use_attention=use_attention,
            reduce_hidden_size=reduce_hidden_size,
            seed=seed,
            eval_buckets=eval_buckets,
        )

    elif model_type == "PROTO":
        model = ProtoModule(
            pretrained_model=pretrained_model,
            label_order_path=all_labels_path,
            num_classes=dataset.get_num_classes(),
            num_training_steps=num_training_steps,
            lr_features=lr_features,
            lr_others=lr_others,
            lr_prototypes=lr_prototypes,
            use_sigmoid=use_sigmoid,
            loss=loss,
            final_layer=final_layer,
            use_prototype_loss=use_prototype_loss,
            num_warmup_steps=num_warmup_steps,
            use_cuda=gpus > 0,
            save_dir=save_dir,
            use_attention=use_attention,
            use_global_attention=use_global_attention,
            dot_product=dot_product,
            normalize=normalize,
            reduce_hidden_size=reduce_hidden_size,
            prototype_vector_path=prototype_vector_path,
            attention_vector_path=attention_vector_path,
            seed=seed,
            eval_buckets=eval_buckets,
        )
    elif model_type == "MULTI_PROTO":
        model = MultiProtoModule(
            pretrained_model=pretrained_model,
            label_order_path=all_labels_path,
            num_classes=dataset.get_num_classes(),
            num_training_steps=num_training_steps,
            lr_features=lr_features,
            lr_others=lr_others,
            lr_prototypes=lr_prototypes,
            use_sigmoid=use_sigmoid,
            loss=loss,
            final_layer=final_layer,
            use_prototype_loss=use_prototype_loss,
            num_warmup_steps=num_warmup_steps,
            use_cuda=gpus > 0,
            save_dir=save_dir,
            use_attention=use_attention,
            dot_product=dot_product,
            normalize=normalize,
            reduce_hidden_size=reduce_hidden_size,
            prototype_vector_path=prototype_vector_path,
            attention_vector_path=attention_vector_path,
            seed=seed,
            eval_buckets=eval_buckets,
            num_prototypes_per_class=num_prototypes_per_class,
            batch_size=batch_size,

        )

    else:
        raise Exception(
            f"{model_type} not found. Please choose a valid model_type.")

    tb_logger = TensorBoardLogger(save_dir, name="lightning_logs")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor=f"val/{metric_opt}",
        mode="max",
        save_last=True,
        save_top_k=1,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="ckpt-{epoch:02d}",
    )

    early_stop_callback = EarlyStopping(
        monitor=f"val/{metric_opt}", patience=patience, mode="max")

    callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]
    if projector_callback:
        embedding_projector_callback = ProjectorCallback(
            dataloader["train"], project_n_batches=project_n_batches
        )
        callbacks.append(embedding_projector_callback)

    if not only_test:
        trainer = pl.Trainer(
            min_steps=15000,
            callbacks=callbacks,
            logger=tb_logger,
            default_root_dir=save_dir,
            devices=gpus,
            check_val_every_n_epoch=check_val_every_n_epoch,
            deterministic=False,
            precision=16,
            accelerator="gpu",
            resume_from_checkpoint=None,)
    else:
        trainer = pl.Trainer(
            min_steps=15000,
            callbacks=callbacks,
            logger=tb_logger,
            default_root_dir=save_dir,
            devices=gpus,
            check_val_every_n_epoch=check_val_every_n_epoch,
            deterministic=False,
            precision=16,
            accelerator="gpu",
            resume_from_checkpoint=pretrained_model_path,)

    if pretrained_model_path:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    if not only_test:
        trainer.fit(model, dataloader["train"], dataloader["val"])
        trainer.test(dataloaders=dataloader["test"], ckpt_path="best")
    else:
        trainer.test(model=model, dataloaders=dataloader["test"])


def start():
    fire.Fire(run_training)
