import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModel

import sproto.utils.utils as utils
from sproto.metrics import metrics


logger = logging.getLogger()


class MultiProtoModule(pl.LightningModule):
    logger: TensorBoardLogger

    def __init__(
        self,
        pretrained_model,
        num_classes,
        label_order_path,
        use_sigmoid=False,
        use_cuda=True,
        lr_prototypes=5e-2,
        lr_features=2e-6,
        lr_others=2e-2,
        num_training_steps=5000,
        num_warmup_steps=1000,
        loss="BCE",
        save_dir="output",
        use_attention=True,
        use_global_attention=False,
        dot_product=False,
        normalize=None,
        final_layer=False,
        reduce_hidden_size=None,
        use_prototype_loss=False,
        prototype_vector_path=None,
        attention_vector_path=None,
        eval_buckets=None,
        seed=7,
        num_prototypes_per_class=1,
        batch_size=10,

    ):
        assert use_attention != use_global_attention, "use_attention and use_global attention cannot be active at the same time"
        super().__init__()
        self.label_order_path = label_order_path
        self.batch_size = batch_size
        self.loss = loss
        self.normalize = normalize
        self.lr_features = lr_features
        self.lr_prototypes = lr_prototypes
        self.lr_others = lr_others
        self.use_sigmoid = use_sigmoid

        self.use_attention = use_attention
        self.use_global_attention = use_global_attention
        self.dot_product = dot_product

        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.save_dir = save_dir
        self.num_classes = num_classes

        self.final_layer = final_layer
        self.use_prototype_loss = use_prototype_loss
        self.prototype_vector_path = prototype_vector_path
        self.eval_buckets = eval_buckets
        self.num_prototypes_per_class_tmp = num_prototypes_per_class

        # ARCHITECTURE SETUP #

        pl.seed_everything(seed=seed)

        # define distance measure
        self.pairwise_dist = nn.PairwiseDistance(p=2)

        # load BERT
        self.bert = AutoModel.from_pretrained(pretrained_model)

        # freeze BERT layers if lr_features == 0
        if lr_features == 0:
            for param in self.bert.parameters():
                param.requires_grad = False

        # define hidden size
        self.hidden_size = self.bert.config.hidden_size
        self.reduce_hidden_size = reduce_hidden_size is not None
        if self.reduce_hidden_size:
            self.reduce_hidden_size = True
            self.bert_hidden_size = self.bert.config.hidden_size
            self.hidden_size = reduce_hidden_size

            # initialize linear layer for dim reduction
            # reset the seed to make sure linear layer is the same as in preprocessing
            pl.seed_everything(seed=seed)
            self.linear = nn.Linear(self.bert_hidden_size, self.hidden_size)

        # load prototype vectors
        if prototype_vector_path is not None:
            prototype_vectors, self.num_prototypes_per_class = self.load_prototype_vectors(
                prototype_vector_path
            )
        elif self.num_prototypes_per_class_tmp > 1:
            prototype_vectors = torch.rand(
                (self.num_prototypes_per_class_tmp, self.num_classes, self.hidden_size))
            num_prototypes_per_class = torch.ones(
                self.num_classes) * self.num_prototypes_per_class_tmp
        else:
            prototype_vectors = torch.rand(
                (self.num_classes, self.hidden_size))
            num_prototypes_per_class = torch.ones(
                self.num_classes)
        self.prototype_vectors = nn.Parameter(
            prototype_vectors, requires_grad=True)

        self.register_buffer('num_prototypes_per_class', num_prototypes_per_class)

        self.prototype_to_class_map = self.build_prototype_to_class_mapping(
            self.num_prototypes_per_class
        )
        self.num_prototypes = self.prototype_to_class_map.shape[0]

        # load attention vectors
        if attention_vector_path is not None:
            attention_vectors = self.load_attention_vectors(
                attention_vector_path)
        elif self.num_prototypes_per_class_tmp > 1:
            attention_vectors = torch.rand(
                (self.num_prototypes_per_class_tmp, self.num_classes, self.hidden_size))
        else:
            attention_vectors = torch.rand(
                (self.num_classes, self.hidden_size))
        self.attention_vectors = nn.Parameter(
            attention_vectors, requires_grad=True)

        if self.final_layer:
            self.final_linear = self.build_final_layer()

        # EVALUATION SETUP #

        # setup metrics
        self.train_metrics = self.setup_metrics()

        # initialise metrics for evaluation on test set
        self.all_metrics = self.train_metrics
        # self.all_metrics = {**self.train_metrics,
        #                     **self.setup_extensive_metrics()}

        torch.proto_vectors = self.prototype_vectors.data.clone().detach()
        torch.attention_vectors = self.attention_vectors.data.clone().detach()

        self.save_hyperparameters()
        logger.info("Finished init.")

    def build_final_layer(self):
        prototype_identity_matrix = torch.zeros(
            self.num_prototypes, self.num_classes)

        for j in range(len(prototype_identity_matrix)):
            prototype_identity_matrix[j, self.prototype_to_class_map[j]] = (
                1.0 /
                self.num_prototypes_per_class[self.prototype_to_class_map[j]]
            )
        return nn.Parameter(prototype_identity_matrix.double(), requires_grad=True)

    def load_prototype_vectors(self, prototypes_per_class_path):
        prototypes_per_class = torch.load(prototypes_per_class_path)

        # store the number of prototypes for each class
        num_prototypes_per_class = torch.tensor(
            [len(prototypes_per_class[key]) for key in prototypes_per_class]
        )

        with open(self.label_order_path) as label_order_file:
            ordered_labels = label_order_file.read().split("\t")

        # get dimension from any of the stored vectors
        vector_dim = len(list(prototypes_per_class.values())[0][0])

        stacked_prototypes_per_class = [
            prototypes_per_class[label]
            if label in prototypes_per_class
            else [np.random.rand(vector_dim)]
            for label in ordered_labels
        ]

        prototype_matrix = torch.tensor(
            [val for sublist in stacked_prototypes_per_class for val in sublist]
        )

        return prototype_matrix, num_prototypes_per_class

    def build_prototype_to_class_mapping(self, num_prototypes_per_class):
        return torch.arange(num_prototypes_per_class.shape[0], device=self.device).repeat_interleave(
            num_prototypes_per_class.long(),
            dim=0)

        # torch.arange(num_prototypes_per_class.shape[0]).repeat_interleave(
        #                                                    num_prototypes_per_class.long(), dim=0)

    def load_attention_vectors(self, attention_vectors_path):
        attention_vectors = torch.load(
            attention_vectors_path, map_location=self.device)

        return attention_vectors

    def setup_metrics(self):

        self.f1 = torchmetrics.F1Score(threshold=0.269)
        self.auroc_micro = metrics.FilteredAUROC(
            num_classes=self.num_classes, compute_on_step=False, average="micro"
        )
        self.auroc_macro = metrics.FilteredAUROC(
            num_classes=self.num_classes, compute_on_step=False, average="macro"
        )
        self.average_precision = torchmetrics.classification.MultilabelAveragePrecision(
            num_labels=self.num_classes, compute_on_step=False, average="macro"
        )

        return {"auroc_micro": self.auroc_micro,
                "auroc_macro": self.auroc_macro,
                "average_precision": self.average_precision,
                }

    def setup_extensive_metrics(self):
        self.pr_curve = metrics.PR_AUC(num_classes=self.num_classes)

        extensive_metrics = {"pr_curve": self.pr_curve}

        if self.eval_buckets:
            buckets = self.eval_buckets

            self.prcurve_0 = metrics.PR_AUCPerBucket(
                bucket=buckets["<5"], num_classes=self.num_classes, compute_on_step=False
            )

            self.prcurve_1 = metrics.PR_AUCPerBdeep

            self.prcurve_2 = metrics.PR_AUCPerBucket(
                bucket=buckets["11-50"], num_classes=self.num_classes, compute_on_step=False
            )

            self.prcurve_3 = metrics.PR_AUCPerBucket(
                bucket=buckets["51-100"], num_classes=self.num_classes, compute_on_step=False
            )

            self.prcurve_4 = metrics.PR_AUCPerBucket(
                bucket=buckets["101-1K"], num_classes=self.num_classes, compute_on_step=False
            )

            self.prcurve_5 = metrics.PR_AUCPerBucket(
                bucket=buckets[">1K"], num_classes=self.num_classes, compute_on_step=False
            )

            self.auroc_macro_0 = metrics.FilteredAUROCPerBucket(
                bucket=buckets["<5"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )
            self.auroc_macro_1 = metrics.FilteredAUROCPerBucket(
                bucket=buckets["5-10"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )
            self.auroc_macro_2 = metrics.FilteredAUROCPerBucket(
                bucket=buckets["11-50"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )
            self.auroc_macro_3 = metrics.FilteredAUROCPerBucket(
                bucket=buckets["51-100"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )
            self.auroc_macro_4 = metrics.FilteredAUROCPerBucket(
                bucket=buckets["101-1K"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )
            self.auroc_macro_5 = metrics.FilteredAUROCPerBucket(
                bucket=buckets[">1K"],
                num_classes=self.num_classes,
                compute_on_step=False,
                average="macro",
            )

            bucket_metrics = {
                "pr_curve_0": self.prcurve_0,
                "pr_curve_1": self.prcurve_1,
                "pr_curve_2": self.prcurve_2,
                "pr_curve_3": self.prcurve_3,
                "pr_curve_4": self.prcurve_4,
                "pr_curve_5": self.prcurve_5,
                "auroc_macro_0": self.auroc_macro_0,
                "auroc_macro_1": self.auroc_macro_1,
                "auroc_macro_2": self.auroc_macro_2,
                "auroc_macro_3": self.auroc_macro_3,
                "auroc_macro_4": self.auroc_macro_4,
                "auroc_macro_5": self.auroc_macro_5,
            }

            extensive_metrics = {**extensive_metrics, **bucket_metrics}

        return extensive_metrics

    def configure_optimizers(self):
        joint_optimizer_specs = [
            {"params": self.prototype_vectors, "lr": self.lr_prototypes},
            {"params": self.attention_vectors, "lr": self.lr_others},
            {"params": self.bert.parameters(), "lr": self.lr_features},
        ]

        if self.final_layer:
            joint_optimizer_specs.append(
                {"params": self.final_linear, "lr": self.lr_prototypes})

        if self.reduce_hidden_size:
            joint_optimizer_specs.append(
                {"params": self.linear.parameters(), "lr": self.lr_others})

        optimizer = torch.optim.AdamW(joint_optimizer_specs)

        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        targets = torch.tensor(batch["targets"], device=self.device)

        if self.use_prototype_loss:
            if batch_idx == 0:
                self.prototype_loss = self.calculate_prototype_loss()
            self.log("prototype_loss", self.prototype_loss,
                     on_epoch=True, batch_size=self.batch_size)

        logits, max_indices, _ = self(batch)

        if self.loss == "BCE":
            train_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target=targets.float()
            )
        else:
            train_loss = torch.nn.MultiLabelSoftMarginLoss()(
                input=torch.sigmoid(logits), target=targets
            )

        self.log("train_loss", train_loss, on_epoch=True,
                 batch_size=self.batch_size)

        if self.use_prototype_loss:
            total_loss = train_loss + self.prototype_loss
        else:
            total_loss = train_loss

        return total_loss

    # @profile
    def forward(self, batch):
        attention_mask = batch["attention_masks"]
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]

        if attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        bert_vectors = bert_output.last_hidden_state

        if self.reduce_hidden_size:
            # apply linear layer to reduce token vector dimension
            token_vectors = self.linear(bert_vectors)
        else:
            token_vectors = bert_vectors

        if self.normalize is not None:
            token_vectors = nn.functional.normalize(
                token_vectors, p=2, dim=self.normalize)

        if self.num_prototypes_per_class_tmp > 1:
            with torch.no_grad():
                logits, max_indices, metadata = self.compute_max_indices(
                    attention_mask, batch, token_vectors)

            logits, max_indices, metadata = self.compute_max_indices(
                attention_mask, batch, token_vectors, max_indices)
        else:
            logits, max_indices, metadata = self.compute_max_indices(
                attention_mask, batch, token_vectors)

        return logits, max_indices, metadata

    # @profile
    def compute_max_indices(self, attention_mask, batch, token_vectors, max_indices=None):
        metadata = None
        if max_indices is not None:
            attention_vectors = torch.take_along_dim(
                self.attention_vectors, max_indices.unsqueeze(-1), dim=0)
            prototype_vectors = torch.take_along_dim(
                self.prototype_vectors, max_indices.unsqueeze(-1), dim=0)
            n_prototypes = 1
        else:
            attention_vectors = self.attention_vectors
            prototype_vectors = self.prototype_vectors
            n_prototypes = self.num_prototypes_per_class_tmp

        if self.use_attention or self.use_global_attention:

            attention_mask_from_tokens = utils.attention_mask_from_tokens(
                attention_mask, batch["tokens"])

            (weighted_samples_per_class,
                attention_per_token_and_class,) = self.calculate_token_class_attention(
                token_vectors,
                attention_vectors,
                n_prototypes,
                mask=attention_mask_from_tokens)

            if self.normalize is not None:
                weighted_samples_per_class = nn.functional.normalize(
                    weighted_samples_per_class, p=2, dim=self.normalize
                )

            if self.num_prototypes_per_class_tmp > 1:
                weighted_samples_per_prototype = weighted_samples_per_class
            else:
                weighted_samples_per_prototype = weighted_samples_per_class.repeat_interleave(
                    self.num_prototypes_per_class.long(), dim=1)

            if self.dot_product:
                score_per_prototype = torch.einsum(
                    "bs,abs->ab", self.prototype_vectors, weighted_samples_per_prototype
                )
            elif max_indices is not None:
                score_per_prototype = -self.pairwise_dist(
                    prototype_vectors,
                    weighted_samples_per_prototype
                ).reshape(-1, self.num_classes)
            elif self.num_prototypes_per_class_tmp > 1:
                permuted = torch.permute(
                    weighted_samples_per_prototype, (1, 0, 2, 3))
                score_per_prototype = -self.pairwise_dist(
                    prototype_vectors.reshape(self.num_prototypes_per_class_tmp *
                                                     self.num_classes, self.hidden_size),
                    permuted.reshape(-1, self.num_prototypes_per_class_tmp *
                                     self.num_classes, self.hidden_size)
                ).reshape(-1, self.num_prototypes_per_class_tmp, self.num_classes)
            elif self.num_prototypes_per_class_tmp == 1:
                score_per_prototype = -self.pairwise_dist(
                    self.prototype_vectors, weighted_samples_per_prototype
                )

            metadata = (
                attention_per_token_and_class,
                weighted_samples_per_prototype,
                score_per_prototype,
                batch["targets"],
                batch["sample_ids"],
            )

        else:
            score_per_prototype = - \
                torch.cdist(token_vectors.mean(dim=1), self.prototype_vectors)
            metadata = None, None, score_per_prototype, batch["targets"], batch["sample_ids"]

        logits, max_indices = self.get_logits_per_class(
            score_per_prototype, max_indices)

        return logits, max_indices, metadata

    # @profile
    def calculate_token_class_attention(self, batch_samples, class_attention_vectors, n_prototypes, mask=None):
        if class_attention_vectors.device != batch_samples.device:
            class_attention_vectors = class_attention_vectors.to(
                batch_samples.device)

        if self.num_prototypes_per_class_tmp == 1:
            score_per_token_and_class = torch.einsum(
                "ikj,mj->imk", batch_samples, class_attention_vectors
            )
        elif self.num_prototypes_per_class_tmp != n_prototypes:
            score_per_token_and_class = torch.einsum(
                "ikj,imj->imk", batch_samples, class_attention_vectors
            )
        else:
            score_per_token_and_class = torch.einsum(
                "ikj,smj->simk", batch_samples, class_attention_vectors
            )

        if self.num_prototypes_per_class_tmp == 1:
            expanded_mask = mask.unsqueeze(dim=1).expand(
                mask.size(0), class_attention_vectors.size(0), mask.size(1))

            expanded_mask = F.pad(input=expanded_mask,
                                  pad=(
                                      0, score_per_token_and_class.shape[2] - expanded_mask.shape[2]),
                                  mode='constant', value=0)

            score_per_token_and_class = score_per_token_and_class.masked_fill(
                (expanded_mask == 0),
                float('-inf'))
        elif self.num_prototypes_per_class_tmp != n_prototypes:
            expanded_mask = mask.unsqueeze(dim=1).expand(
                mask.size(0), class_attention_vectors.size(1), mask.size(1))
            score_per_token_and_class = score_per_token_and_class.masked_fill(
                (expanded_mask == 0),
                float('-inf'))
        else:
            expanded_mask = mask.unsqueeze(dim=1).expand(self.num_prototypes_per_class_tmp, mask.size(
                0), class_attention_vectors.size(1), mask.size(1))
            score_per_token_and_class = score_per_token_and_class.masked_fill(
                (expanded_mask == 0),
                float('-inf'))

        if self.use_sigmoid:
            attention_per_token_and_class = (
                torch.sigmoid(score_per_token_and_class) /
                score_per_token_and_class.shape[2]
            )
        else:
            if self.num_prototypes_per_class_tmp == 1:
                attention_per_token_and_class = F.softmax(
                    score_per_token_and_class, dim=2)
            elif self.num_prototypes_per_class_tmp != n_prototypes:
                attention_per_token_and_class = F.softmax(
                    score_per_token_and_class, dim=2)
            else:
                attention_per_token_and_class = F.softmax(
                    score_per_token_and_class, dim=3)

        if self.num_prototypes_per_class_tmp == 1 or self.num_prototypes_per_class_tmp != n_prototypes:
            weighted_samples_per_class = torch.einsum('ikjm,ikj->ikm',
                                                 batch_samples.unsqueeze(dim=1).expand(batch_samples.size(0), self.num_classes, batch_samples.size( 1), batch_samples.size(2)),
                                                 attention_per_token_and_class)

        else:
            expanded = batch_samples.unsqueeze(dim=1).expand(
                batch_samples.size(0),
                self.num_classes,
                batch_samples.size(1),
                batch_samples.size(2),
            )
            weighted_samples_per_class = torch.einsum("ikjm,sikj->sikm", expanded, attention_per_token_and_class)

        return weighted_samples_per_class, attention_per_token_and_class

    # @profile
    def get_logits_per_class(self, score_per_prototype, max_indices=None):
        if self.final_layer:
            if score_per_prototype.device != self.final_linear.device:
                score_per_prototype = score_per_prototype.to(
                    self.final_linear.device)

            return torch.matmul(score_per_prototype, self.final_linear)

        else:
            if self.num_prototypes_per_class_tmp == 1:
                max_logits_per_class = score_per_prototype
                max_logits_per_class_index = None
            elif max_indices is not None:
                return score_per_prototype, max_indices
            else:
                max_logits_per_class, max_logits_per_class_index = torch.max(
                    score_per_prototype, dim=1)
            return max_logits_per_class, max_logits_per_class_index

    def calculate_prototype_loss(self):
        prototype_loss = (
            100
            / torch.tensor(
                [
                    torch.cdist(
                        self.prototype_vectors[
                            (self.prototype_to_class_map == i).nonzero().flatten()
                        ][:1],
                        self.prototype_vectors[
                            (self.prototype_to_class_map == i).nonzero().flatten()
                        ][1:],
                    ).min()
                    for i in range(self.num_classes)
                    if len((self.prototype_to_class_map == i).nonzero()) > 1
                ]
            ).sum()
        )
        return prototype_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            targets = torch.tensor(batch["targets"], device=self.device)

            logits, max_indices, _ = self(batch)

            for metric_name in self.train_metrics:
                metric = self.train_metrics[metric_name]
                metric(torch.sigmoid(logits), targets)

    def validation_epoch_end(self, outputs) -> None:
        for metric_name in self.train_metrics:
            metric = self.train_metrics[metric_name]
            self.log(f"val/{metric_name}", metric.compute(),
                     batch_size=self.batch_size)
            metric.reset()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            targets = torch.tensor(batch["targets"], device=self.device)

            logits, max_indices, _ = self(batch)
            preds = torch.sigmoid(logits)
            del logits
            for metric_name in self.all_metrics:
                metric = self.all_metrics[metric_name]
                metric(preds, targets)

        return preds, targets

    def test_epoch_end(self, outputs) -> None:
        log_dir = self.logger.log_dir
        for metric_name in self.all_metrics:
            metric = self.all_metrics[metric_name]
            value = metric.compute()
            self.log(f"test/{metric_name}", value, batch_size=self.batch_size)

            with open(os.path.join(log_dir, "test_metrics.txt"), "a") as metrics_file:
                metrics_file.write(f"{metric_name}: {value}\n")

            metric.reset()

        predictions = torch.cat([out[0] for out in outputs])

        targets = torch.cat([out[1] for out in outputs])

        pr_auc = metrics.calculate_pr_auc(
            prediction=predictions, target=targets, num_classes=self.num_classes, device=self.device
        )

        with open(os.path.join(self.logger.log_dir, "PR_AUC_score.txt"), "w") as metrics_file:
            metrics_file.write(f"PR AUC: {pr_auc.cpu().numpy()}\n")
