import torch


def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.
    ## the tokenizer can only take a single batch at a time, so split a dataframe up row by row within a for loop?
    ## note that the function in the lab notebook is taking the example batch as a dictionary, here it is a dataframe
    ## the tokenizer will return input_ids and attention_mask as two lists, and we need to collection of input ids and
    ## attention masks as tensors
    ## separate the situations where a sequence need to be padded or need to be truncated
    tokenized_data = tokenizer(
            list(dataset['question']), 
            list(dataset['passage']), 
            truncation=True, 
            max_length = max_seq_length,
            padding = "max_length",
            return_offsets_mapping=True,
      )
    input_ids_list = tokenized_data['input_ids']
    attention_mask_list = tokenized_data['attention_mask']
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
    return input_ids, attention_mask


def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.
    labels = []
    for l in dataset['label']:
      if l:
        labels.append(1)
      else:
        labels.append(0)
    return labels