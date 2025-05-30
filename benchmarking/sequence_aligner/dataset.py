from typing import List, Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from typing_extensions import TypedDict

from sequence_aligner.containers import TrainingExample,PredictExample
from sequence_aligner.labelset import LabelSet


class ExpectedAnnotationShape(TypedDict):
    start: int
    end: int
    label: str


class ExpectedDataItemShape(TypedDict):
    content: str  # The Text to be annotated
    annotations: List[ExpectedAnnotationShape]
    
class PredictDatasetBySeq(Dataset):
    def __init__(
            self,
            data: Any,
            label_set: LabelSet,
            tokenizer: PreTrainedTokenizerFast,
            tokens_per_batch=32,
            window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        
        news = data
        self.texts = news
        
        
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.pred_examples: List[PredictExample] = []
        for encoding in tokenized_batch.encodings:
            length = len(encoding)  # How long is this sequence
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                # print("padding_to_add",padding_to_add)
                # print("==",label[start:end])
                # print( "label:",label[start:end]
                #                 + [0] * padding_to_add )
                self.pred_examples.append(
                    PredictExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                                  + [self.tokenizer.pad_token_id]
                                  * padding_to_add,  # padding if needed
                        attention_masks=(
                                encoding.attention_mask[start:end]
                                + [0]
                                * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )
                
    def __len__(self):
        return len(self.pred_examples)

    def __getitem__(self, idx) -> PredictExample:

        return self.pred_examples[idx]
    
        
class PredictDatasetCRF(Dataset):
    def __init__(
            self,
            data: Any,
            label_set: LabelSet,
            tokenizer: PreTrainedTokenizerFast,
            tokens_per_batch=32,
            window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        
        news = data["content"]
        self.texts = news
        
        
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.pred_examples: List[PredictExample] = []
        for encoding in tokenized_batch.encodings:
            length = len(encoding)  # How long is this sequence
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                # print("padding_to_add",padding_to_add)
                # print("==",label[start:end])
                # print( "label:",label[start:end]
                #                 + [0] * padding_to_add )
                self.pred_examples.append(
                    PredictExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                                  + [self.tokenizer.pad_token_id]
                                  * padding_to_add,  # padding if needed
                        attention_masks=(
                                encoding.attention_mask[start:end]
                                + [0]
                                * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )
                
    def __len__(self):
        return len(self.pred_examples)

    def __getitem__(self, idx) -> PredictExample:

        return self.pred_examples[idx]
    
class TrainingDatasetCRF(Dataset):
    def __init__(
            self,
            data: Any,
            label_set: LabelSet,
            tokenizer: PreTrainedTokenizerFast,
            tokens_per_batch=32,
            window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )

            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT
        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                # print("padding_to_add",padding_to_add)
                # print("==",label[start:end])
                # print( "label:",label[start:end]
                #                 + [0] * padding_to_add )
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                                  + [self.tokenizer.pad_token_id]
                                  * padding_to_add,  # padding if needed
                        labels=(
                                label[start:end]
                                + [-1] * padding_to_add  # padding if needed
                        ), 
                        attention_masks=(
                                encoding.attention_mask[start:end]
                                + [0]
                                * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )
    
    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:

        return self.training_examples[idx]


class TrainingDataset(Dataset):
    def __init__(
            self,
            data: Any,
            label_set: LabelSet,
            tokenizer: PreTrainedTokenizerFast,
            tokens_per_batch=32,
            window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )

            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT
        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                # print("padding_to_add",padding_to_add)
                # print("==",label[start:end])
                # print( "label:",label[start:end]
                #                 + [0] * padding_to_add )
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                                  + [self.tokenizer.pad_token_id]
                                  * padding_to_add,  # padding if needed
                        labels=(
                                label[start:end]
                                + [-100] * padding_to_add  # padding if needed
                        ), 
                        attention_masks=(
                                encoding.attention_mask[start:end]
                                + [0]
                                * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )
    
    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:

        return self.training_examples[idx]
