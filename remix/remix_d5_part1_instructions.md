
# REMIX Day 5, Part 1 - IOIDataset

There is a lot of bookkeeping involved in running the IOI experiments, which we've provided for you. The class `IOIDataset` handles all the dataset related computation including tokenization and computation of the relevant indices.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Readings](#readings)
- [Constructor](#constructor)
- [Tokenization](#tokenization)
- [Word_idx](#wordidx)
- [Metadata](#metadata)
- [Flips](#flips)
- [Composing Flips](#composing-flips)
- [Generating Your Own Flip](#generating-your-own-flip)

## Learning Objectives

After going through this material, you should be able to:

- Use the IOIDataset API
- Understanding the principle to create similar datasets during your future project

## Readings

* The [slides from the lecture](https://docs.google.com/presentation/d/13Bvmo8E6N5qhgj1yCXq5O7zNRzNNXZLzexlgdzdgZ_E/edit?usp=sharing)
* [A guide to language model interpretability](https://docs.google.com/document/d/1cSdLwC9mVaLxMDKaXbOsxrglwATOjc0NfMuUvxLNnNE/edit?usp=sharing) (most of the content covered in the lecture, here to refer to it if needed)




```python
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
from copy import deepcopy
from remix_d5_utils import IOIDataset

MAIN = __name__ == "__main__"
if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

```

## Constructor

The simplest way to construct an `IOIDataset` involves two arguments: the number of prompts to generate, and a string representing the type of prompt.

Exercise: Try out the different types of prompt. You should be able to Ctrl+Click the IOIDataset below and see the legal string values. Or, if you put an illegal prompt type then your IDE should complain and show you a message with the legal string values.

Exercise: Is IOIDataset deterministic, or are the prompts random each time?


```python
ioi_dataset = IOIDataset(3, prompt_type="mixed", seed=78)
print("Prompts: ", ioi_dataset.prompts_text)
print("Tokens: ", ioi_dataset.prompts_toks)

```

## Tokenization

Notice that the sentence begins with `<|endoftext|>`. The tokenizer recognizes this special string and replaces it with a specific token 50256. During the training of GPT2, we believe there were no padding tokens and two different articles could appear within one training example, separated by this token.

So the idea behind putting this at the start is to mimic GPT2's training, where the thing after this token comes at the start of a new article.

Not using '<|endoftext|>' token lead to big difference if the first token of the sentence is a name (e.g. "Alice and Bob ..." vs "<|endoftext|> Alice and Bob"). Results are similar when the first token is not important (e.g. "Then, Alice and Bob ..." vs "<|endoftext|>Then, Alice and Bob").

We use words for names, places and objects that are single tokens. This makes it easier to study the model: a token position contains all the information about a given name instead of being split between two for instance.

## Word_idx

This variable of type `Dict[str,torch.Tensor]` is dictionary that maps the name of a word to its index in each of the prompts. For example, `word_idx["IO"]` will give you a tensor of ints of shape `(NB_PROMPTS)`. Each entry is the index of the IO token in the prompt. The possible keys are "IO", "S1", "S2", "S1+1" and "END" for the index of the last token (" to").



```python
print(ioi_dataset.prompts_text[0])
for (k, v) in ioi_dataset.word_idx.items():
    print(f" The token {k} is at position {v[0]}")

```

To check that the position are correct, we can use the `prompts_text_toks` that store the tokenized prompt as a list of string. The names are replaced by their semantic annotation (IO, S1 or S2). This is also helpful to know how sentences are tokenized.

TBD: this should probably be a docstring on the actual class's field


```python
ioi_dataset.prompts_text_toks[0]

```


`prompt_metadata`: a list of dictionaries containing metadata about the prompts. They include the string of the placeholders, the id of the template used and the order of the names.


```python
print(f"Some metadata about '{ioi_dataset.prompts_text[0]}'")
print()
print(ioi_dataset.prompts_metadata[0])

```

## Metadata

You can use metadata to create copies (or modification) of the dataset. In this case, there is no randomness involved: all the information needed to create the dataset is contained in the metadata.



```python
new_metadata = deepcopy(ioi_dataset.prompts_metadata)
new_metadata[0]["S"] = "Robert"
new_ioi_dataset = IOIDataset(N=ioi_dataset.N, prompt_type=ioi_dataset.prompt_type, manual_metadata=new_metadata)
print(f"Original prompt: {ioi_dataset.prompts_text[0]}")
print(f"New prompt: {new_ioi_dataset.prompts_text[0]}")

```

## Flips

By "flip", we mean replacing part of a prompt with a new random name. For instance:


```python
flipped_io_dataset = ioi_dataset.gen_flipped_prompts("IO")
print("Original: ", ioi_dataset.prompts_text[0])
print("Flipped: ", flipped_io_dataset.prompts_text[0])

```

We can also flip the S1 token to a random name. By doing so, we change the name family! The new dataset is part of the ABC family and not IOI as the sentences now contains three distinct names.

This means:

- The word_idx has different keys. "IO1" is the old "IO" and "IO2" is the newly created IO taking the place of "S1". "S" is the old "S2". In particular, this means that the number of the IO doesn't refer to their position in the sequence, you can have IO2 appearing before IO1.
- `prompt_metadata` now contains extra key "IO2" that is the value of the newly created IO, and "IO" is removed.

The majority of the keys for the ABC and IOI family are different to force you to know what you are looking at. After composing several filp, it's easy to forget if you are looking at a ABC or IOI dataset.


```python
flipped_s1_dataset = ioi_dataset.gen_flipped_prompts("S1")
print(f"Original prompt: {ioi_dataset.prompts_text[0]}")
print(f"New prompt: {flipped_s1_dataset.prompts_text[0]}")
assert flipped_s1_dataset.prompt_family == "ABC"
print(flipped_s1_dataset.word_idx)

```

## Composing Flips

Naturally, you can do multiple flips and eventually reach a dataset that has nothing in common with the original.


```python
two_flip = flipped_s1_dataset.gen_flipped_prompts("IO1")
print(f"Original prompt: {ioi_dataset.prompts_text[0]}")
print(f"After fliping S1 then IO1: {two_flip.prompts_text[0]}")
three_flip = two_flip.gen_flipped_prompts("S")
print(f"Original prompt: {ioi_dataset.prompts_text[0]}")
print(f"After fliping S1, IO1 and S: {three_flip.prompts_text[0]}")

```

## Generating Your Own Flip

We want to modified the sentences such that the two first names appearing in each sentences are flipped. For instance, "Alice and Bob ..." becomes "Bob and Alice ...".

Exercise: implement `order_flip` in the case where the dataset is from the family ABC. (The case of IOI is a bit tricky and involve template manipulation.).

<details>
<summary>Click here for a hint</summary>
You can flip the values of IO1 and IO2 in the metadata and create a new dataset using manual_metadata.
</details>



```python
def order_flip(dataset: IOIDataset) -> IOIDataset:
    """
    - For a dataset from the ABC family, generate a new dataset where the two first names appears in flipped order. "Alice and Bob ..." becomes "Bob and Alice ...".
    """
    assert dataset.prompt_family == "ABC"
    new_prompts_metadata = deepcopy(dataset.prompts_metadata)
    "TODO: YOUR CODE HERE"
    pass


original = IOIDataset(3, prompt_type="mixed")
original = original.gen_flipped_prompts("S2")
flipped_order = order_flip(original)
assert flipped_order.prompt_family == "ABC"
for i in range(len(flipped_order)):
    assert flipped_order.prompts_metadata[i]["IO2"] == original.prompts_metadata[i]["IO1"]
    assert flipped_order.prompts_metadata[i]["IO1"] == original.prompts_metadata[i]["IO2"]
    assert flipped_order.prompts_metadata[i]["S"] == original.prompts_metadata[i]["S"]

```

Exercise: why does the following tests fail?
<details>
<summary>Answer</summary>
The _values_ of IO1 and IO2 where changed, but the position of the words labeled IO1 and IO2 are the same. The word_idx is a mapping from a token labeled by its role (IO, S, IO2 etc) to its position. So the word_idx is not changed.
</details>


```python
print(flipped_order.word_idx["IO1"], original.word_idx["IO2"])
assert (flipped_order.word_idx["IO1"] == original.word_idx["IO2"]).all()
assert (flipped_order.word_idx["IO2"] == original.word_idx["IO1"]).all()

```