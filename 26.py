#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP_full_release

For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf

Authors
 * Yingzhi WANG 2021
"""

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import torch.nn as nn
import os
import sys
import yaml
import time
import torch
import shutil
import logging
import inspect
import pathlib
import argparse
import tempfile
import speechbrain as sb
from datetime import date
from enum import Enum, auto
from tqdm.contrib import tqdm
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.nn import DataParallel as DP
from torch.utils.data import IterableDataset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from hyperpyyaml import resolve_references
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DistributedSamplerWrapper
from speechbrain.dataio.sampler import ReproducibleRandomSampler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
import os

# CUDA_VISIBLE_DEVICES=1 python train_aware.py hparams/train_aware.yaml
# CUDA_VISIBLE_DEVICES=1,2 python train_aware.py hparams/train_aware.yaml --data_parallel_backend

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# device = torch.device("cuda")


class SpeakerBrain(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        x_e, lens1 = batch.sig_e
        x_t, lens2 = batch.sig_t
        
#         print(self.hparams.lr_annealing(epoch))
        
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
        # Embeddings + speaker classifier
#         emb_e = self.modules.embedding_model(feats)
        h_e = classifier.encode_batch(x_e)
#         print("*************", h_e.shape) # [32, 1, 192]  [32, 1, 96]
        outputs_e = self.modules.asp_pool(h_e.transpose(1, 2), lens1)
        outputs_e = outputs_e.view(outputs_e.shape[0], -1)
#         print("************* outputs_e:", outputs_e.shape)  # torch.Size([32, 384])   [32, 192]
        emb_e = self.modules.dense1(outputs_e)   # [32, 512]
        relu = torch.nn.ReLU()    
        batch_norm3 = self.modules.batchnorm3
        emb_e = batch_norm3(relu(emb_e))
        
        drop = torch.nn.Dropout1d(p=0.3)
        outputs_o=drop(emb_e) 
#         print("************* emb_e:", emb_e.shape)  # torch.Size([32, 1, 512])   
        outputs_e = self.modules.dense2(emb_e)
#         print("************* outputs_e:", outputs_e.shape)  # torch.Size([32, 1, n])  
        

    
    # 2. speaker aware
 

        # Embeddings + speaker classifier
        h_t = classifier.encode_batch(x_t)
#         h_t = self.modules.embedding_model(feats)    # [32, 1, 192]  [32, 1, 96]
        en = self.modules.fc1(emb_e) 
#         print("************* en:", en.shape)  # en: torch.Size([32, 1, 512])
        outputs_t = self.modules.fc2(h_t)   # torch.Size([32, 1, 192])  [32, 1, 96]
#         print("************* outputs_t:", outputs_t.shape) # torch.Size([32, 1, 192])  [32, 1, 96]
        en = torch.unsqueeze(en, 1) # torch.Size([32, 1, 512])
#         en = en.repeat(1,149,1)   # torch.Size([32, 149, 512])
        outputs_o = torch.concat((outputs_t, en),2)
#         print("************* outputs_o:", outputs_o.shape)   # torch.Size([32, 1, 704])   [32, 1, 608]
        outputs_o = self.modules.bn_fc1(outputs_o)   # [32, 1, 352]   [32, 1, 304]
        batch_norm1 = self.modules.batchnorm1
        outputs_o = batch_norm1(relu(outputs_o)) 
        
        drop = torch.nn.Dropout1d(p=0.3)
        outputs_o=drop(outputs_o)   # added dropout before dense ayers
        
        outputs_o = self.modules.bn_fc2(outputs_o)  # [32, 1, 256]   [32, 1,128]
        batch_norm2 = self.modules.batchnorm2
        outputs_o = batch_norm2(relu(outputs_o)) #   
        
        drop = torch.nn.Dropout1d(p=0.3)
        outputs_o=drop(outputs_o) 
        
        outputs_s = self.modules.bn_fc3(outputs_o)  #   [32, 1, 192]   [32, 1, 96]   
#         print("************* outputs_s:", outputs_s.shape)
        outputs_m = torch.sigmoid(outputs_s)        
        outputs_mm = torch.mul(h_t, outputs_m)  # 
#         print("************* outputs_mm:", outputs_mm.shape) #   [32, 1, 192]   
        outputs_asp = self.modules.asp_pool(outputs_mm.transpose(1, 2), lens1)
        outputs_t = outputs_asp.view(outputs_asp.shape[0], -1)  
#         print("************* outputs_t:", outputs_t.shape)  # torch.Size([32, 384])  ([32, 192])
        emb_t = self.modules.dense1(outputs_t)    # 32x384   32x192
        batch_norm3 = self.modules.batchnorm3
        emb_t = batch_norm3(relu(emb_t))
        drop = torch.nn.Dropout1d(p=0.3)
        outputs_t=drop(emb_t) 
        outputs_t = self.modules.dense2(emb_t)


        """if you use softmax"""
        outputs_e = self.hparams.log_softmax(outputs_e)
        outputs_t = self.hparams.log_softmax(outputs_t)

        return [outputs_e, outputs_t]
#         return [emb_e, emb_t]
        


    def compute_objectives(self, outputs, batch, stage):
        """Computes the loss using speaker-id as label.
        """
#         print(self.optimizer)
        label_e = batch.label_encoded_e[0]     
        label_t = batch.label_encoded_t[0]

#         log_prob = sb.nnet.losses.LogSoftmaxWrapper(sb.nnet.losses.AdditiveAngularMargin(margin=0.2, scale=32))
#         """to meet the input form of nll loss"""
# #         emoid = emoid.squeeze(1)
#         one_hot_e = torch.nn.functional.one_hot(label_e.squeeze(1), num_classes = 6295)
#         one_hot_t = torch.nn.functional.one_hot(label_t.squeeze(1), num_classes = 6295)
#         loss_e = log_prob(outputs[0], label_e)
#         loss_t = log_prob(outputs[1], label_t)
#         if stage != sb.Stage.TRAIN:
#             predictions_e = self.hparams.AdditiveAngularMargin(outputs[0], one_hot_e)
#             predictions_t = self.hparams.AdditiveAngularMargin(outputs[1], one_hot_t)
# #             self.error_metrics.append(batch.id, predictions, emoid)
# #             print(torch.concat((predictions_e, predictions_t),0).shape, torch.concat((label_e, label_t),0).shape)
#             self.error_metrics.append(batch.id, predictions_e, label_e.squeeze(1))
#             self.error_metrics.append(batch.id, predictions_t, label_t.squeeze(1))


#         return loss_e + loss_t


        """2. if you use softmax, but with traditinoal dataio"""
        loss_e = self.hparams.compute_cost_nll(outputs[0], label_e.squeeze(1))
        loss_t = self.hparams.compute_cost_nll(outputs[1], label_t.squeeze(1))

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, outputs[0], label_e.squeeze(1))
            self.error_metrics.append(batch.id, outputs[1], label_t.squeeze(1))

        return loss_e + loss_t



#     def fit_batch(self, batch):
#         """Trains the parameters given a single batch in input"""
#         outputs = self.compute_forward(batch, sb.Stage.TRAIN)
#         loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
#         loss.backward()
#         if self.check_gradients(loss):
# #             self.wav2vec2_optimizer.step()
#             self.optimizer.step()

# #         self.wav2vec2_optimizer.zero_grad()
#         self.optimizer.zero_grad()

#         return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

#         # Set up statistics trackers for this stage
#         self.loss_metric = sb.utils.metric_stats.MetricStats(
# #             metric=sb.nnet.losses.AdditiveAngularMargin
#             metric=sb.nnet.losses.nll_loss
#         )
        
#         if stage == sb.Stage.TRAIN:
#             print("******** RESET optimizer at the beginning of each epoch: ")
# #             self.optimizer.__setstate__({'state': defaultdict(dict)})
#             self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())
#         print(self.optimizer)
#         self.checkpointer.add_recoverable("optimizer", self.optimizer)

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
        print(self.optimizer)

        
    def transcribe_dataset(
        self,
        dataset, # Must be obtained from the dataio_function
        min_key, # We load the model with the lowest WER
        loader_kwargs # opts for the dataloading
    ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

#         # Now we iterate over the dataset and we simply compute_forward and decode
#         with torch.no_grad():

#             pred = []
#             actual = []
#             for batch in tqdm(dataset, dynamic_ncols=True):

#                 # Make sure that your compute_forward returns the predictions !!!
#                 # In the case of the template, when stage = TEST, a beam search is applied 
#                 # in compute_forward(). 
#                 out = self.compute_forward(batch, stage=sb.Stage.TEST)
#                 out_e = out[0]
#                 print("***************", out_e.shape)
#                 out_t = out[1]
#                 print("***************", out_t.shape)
#                 break
#                 label_e = batch.label_encoded_e[0]
#                 label_t = batch.label_encoded_t[0]
#                 actual.extend(label_e)
#                 actual.extend(label_t)
                
#                 tag_prob_e = F.softmax(out_e, dim = 1)
#                 tag_prob_t = F.softmax(out_t, dim = 1)

#                 tag_class_e = torch.argmax(tag_prob_e, dim=1).cpu()
#                 tag_class_t = torch.argmax(tag_prob_t, dim=1).cpu()

#                 pred.extend(tag_class_e)
#                 pred.extend(tag_class_t)


#         return pred, actual


#         # for vox2 / in-domain
#         similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
#         with torch.no_grad():

#             le = []
#             lt = []
#             scores = []
#             for batch in tqdm(dataset, dynamic_ncols=True):

#                 # Make sure that your compute_forward returns the predictions !!!
#                 # In the case of the template, when stage = TEST, a beam search is applied 
#                 # in compute_forward(). 
#                 out = self.compute_forward(batch, stage=sb.Stage.TEST)
#                 out_e = out[0]
# #                 print("***************", out_e.shape)
#                 out_t = out[1]
# #                 print("***************", out_t.shape)
#                 sc = similarity(out_e, out_t)
# #                 break
#                 label_e = batch.label_encoded_e[0]
#                 label_t = batch.label_encoded_t[0]
#                 le.extend(label_e)
#                 lt.extend(label_t)
#                 scores.extend(sc)

#         return le, lt, scores



        # for vox1/ami
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        with torch.no_grad():

            names = []
            scores = []
            for batch in tqdm(dataset, dynamic_ncols=True):

                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                out_e = out[0]
                out_t = out[1]
                sc = similarity(out_e, out_t)
                name = batch.id
                names.extend(name)
                    
                scores.extend(sc)



        return names, scores





#         # For audioband
#         similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
#         with torch.no_grad():

#             scores = []
#             ids = []
#             for batch in tqdm(dataset, dynamic_ncols=True):

#                 # Make sure that your compute_forward returns the predictions !!!
#                 # In the case of the template, when stage = TEST, a beam search is applied 
#                 # in compute_forward(). 
#                 out = self.compute_forward(batch, stage=sb.Stage.TEST)
#                 out_e = out[0]
# #                 print("***************", out_e.shape)
#                 out_t = out[1]
# #                 print("***************", out_t.shape)
#                 sc = similarity(out_e, out_t)

#                 scores.extend(sc)
#                 ids.extend(batch.id)
# #                 print("***************", batch.id[0])
# #                 break



#         return ids, scores


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav1","wav2")
    @sb.utils.data_pipeline.provides("sig_e","sig_t")
    def audio_pipeline(wav1, wav2):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig_e = sb.dataio.dataio.read_audio(wav1)
        sig_t = sb.dataio.dataio.read_audio(wav2)
        yield sig_e
        yield sig_t
    
#         # Define audio pipeline
#     @sb.utils.data_pipeline.takes("wav2")
#     @sb.utils.data_pipeline.provides("sig_t")
#     def audio_pipeline(wav2):
#         """Load the signal, and pass it and its length to the corruption class.
#         This is done on the CPU in the `collate_fn`."""
#         sig_t = sb.dataio.dataio.read_audio(wav2)
#         return sig_t

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.add_unk()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("label1","label2")
    @sb.utils.data_pipeline.provides("label_e", "label_encoded_e","label_t","label_encoded_t")   # IN ORDER
    def label_pipeline(label1, label2):
        label_e = label1
        yield label_e
        label_encoded_e = label_encoder.encode_label_torch(label1)
        yield label_encoded_e
        label_t = label2
        yield label_t
        label_encoded_t = label_encoder.encode_label_torch(label2)
        yield label_encoded_t
        
#     # Define label pipeline:
#     @sb.utils.data_pipeline.takes("label2")
#     @sb.utils.data_pipeline.provides("label_t", "label_encoded_t")
#     def label_pipeline(label2):
#         label_t = label2
#         yield label_t
#         label_encoded_t = label_encoder.encode_label_torch(label2)
#         yield label_encoded_t

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            dynamic_items=[audio_pipeline, label_pipeline],
#             output_keys=["id", "sig", "label_encoded"],
            output_keys=["id", "sig_e","sig_t", "label_encoded_e", "label_encoded_t"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="label_e"  # means to encode label_e
    )

    return datasets





# RECIPE BEGINS!
if __name__ == "__main__":
    

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

#     hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
#     # freeze the feature extractor part when unfreezing
#     if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
#         hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    
    

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

#     # Load the best checkpoint for evaluation
#     test_stats = emo_id_brain.evaluate(
#         test_set=datasets["test"],
#         min_key="error_rate",
#         test_loader_kwargs=hparams["dataloader_options"],
#     )
    
#     pred, act = emo_id_brain.transcribe_dataset(
#         dataset=datasets["test"], # Must be obtained from the dataio_function
#         min_key="error_rate", # We load the model with the lowest WER
#         loader_kwargs=hparams["dataloader_options"], # opts for the dataloading
#     )

#     le, lt, scores = emo_id_brain.transcribe_dataset(
#         dataset=datasets["test"], # Must be obtained from the dataio_function
#         min_key="error_rate", # We load the model with the lowest WER
#         loader_kwargs=hparams["dataloader_options"], # opts for the dataloading
#     )

#     ids, scores = emo_id_brain.transcribe_dataset(
#         dataset=datasets["test"], # Must be obtained from the dataio_function
#         min_key="error_rate", # We load the model with the lowest WER
#         loader_kwargs=hparams["dataloader_options"], # opts for the dataloading
#     )
    
#     import pandas as pd

#     #create DataFrame
#     df = pd.DataFrame(columns=['actual', 'pred'])
#     df['actual'] = pd.Series(torch.Tensor(act).cpu())
#     df['pred'] = pd.Series(torch.Tensor(pred).cpu())

#     df.to_csv("details_tmp.csv")

#     import pandas as pd

#     #create DataFrame
#     df = pd.DataFrame(columns=['enroll', 'test', 'score'])
#     df['enroll'] = pd.Series(torch.Tensor(le).cpu())
#     df['test'] = pd.Series(torch.Tensor(lt).cpu())
#     df['score'] = pd.Series(torch.Tensor(scores).cpu())

#     df.to_csv("vox2_test.csv")


#     import pandas as pd

#     #create DataFrame
#     df = pd.DataFrame(columns=['ids', 'score'])
#     df['ids'] = pd.Series(ids)
#     df['score'] = pd.Series(torch.Tensor(scores).cpu())

#     df.to_csv("ami_test_vox1.csv")
    