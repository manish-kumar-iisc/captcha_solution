import os
import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
import pdb
import numpy as np

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# Define paths
DATA_DIR = "/home/manish/test_1/dataset/dataset/train-images/train-images/"  # Update this path
CSV_FILE = "/home/manish/test_1/dataset/dataset/captcha_data.csv"  # Update this path

# Load CSV and ensure labels are 6-digit strings
df = pd.read_csv(CSV_FILE)
df=df.head(1000)
df["label"] = df["solution"].apply(lambda x: str(x).zfill(6))
df.drop(columns=["solution"])

# Initialize Processor (tokenizer + image processor)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=8):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['image_path_1'][idx]
        text = self.df['label'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

train_ds = IAMDataset(root_dir=DATA_DIR,
                           df=df,
                           processor=processor)
test_ds = IAMDataset(root_dir=DATA_DIR,
                           df=df,
                           processor=processor)


print("Number of training examples:", len(train_ds))
encoding = train_ds[0]
for k,v in encoding.items():
  print(k, v.shape)


from transformers import VisionEncoderDecoderModel

# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 10
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 1
model.config.length_penalty = 2.0
model.config.num_beams = 1

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=False, 
    output_dir="./",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
)

from datasets import load_metric

cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=default_data_collator,
)
pdb.set_trace()

trainer.train()
model.save_pretrained("./fine_tuned_trocr")
processor.save_pretrained("./fine_tuned_trocr")

# Load model for inference
processor = TrOCRProcessor.from_pretrained("./fine_tuned_trocr")
model = VisionEncoderDecoderModel.from_pretrained("./fine_tuned_trocr")

def predict_captcha(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(pixel_values)

    pred_text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return pred_text

# Example inference
image_path = "/home/manish/test_1/dataset/dataset/train-images/train-images/image_train_345.png"  # Update this path
print("Predicted CAPTCHA:", predict_captcha(image_path))

