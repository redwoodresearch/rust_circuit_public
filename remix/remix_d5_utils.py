import torch
from copy import deepcopy

from typing import Union, Literal, Any
import random
from interp.tools.interpretability_tools import get_interp_tokenizer
import rust_circuit as rc


from interp.circuit.interop_rust.algebric_rewrite import split_to_concat
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.interpretability_tools import print_max_min_by_tok_k_torch

from interp.tools.indexer import TORCH_INDEXER as I

from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import (
    logprob_on_labels,
    make_arr,
    match_nodes_except,
    CopyDsSampler,
    direct_path_patching_up_to,
)
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    InterpNode,
    ExactSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.circuit_model_rewrites import (
    ALL_BASIC_NAMES,
    AttnSuffixForGpt,
    HeadOrMlpType,
    MLPHeadAndPosSpec,
    split_heads_and_positions,
)


ABBA_TEMPLATES = [
    "<|endoftext|>Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
]
#     "<|endoftext|>Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
# "<|endoftext|>Then, {IO} and {S1} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to",
# "<|endoftext|>Then, {IO} and {S1} were working at the {PLACE}. {S2} decided to give a {OBJECT} to",
# "<|endoftext|>Then, {IO} and {S1} were thinking about going to the {PLACE}. {S2} wanted to give a {OBJECT} to",
# "<|endoftext|>Then, {IO} and {S1} had a long argument, and afterwards {S2} said to",
# "<|endoftext|>After {IO} and {S1} went to the {PLACE}, {S2} gave a {OBJECT} to",
# "<|endoftext|>When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give it to",
# "<|endoftext|>When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give the {OBJECT} to",
# "<|endoftext|>While {IO} and {S1} were working at the {PLACE}, {S2} gave a {OBJECT} to",
# "<|endoftext|>While {IO} and {S1} were commuting to the {PLACE}, {S2} gave a {OBJECT} to",
# "<|endoftext|>After the lunch, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
# "<|endoftext|>Afterwards, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
# "<|endoftext|>Then, {IO} and {S1} had a long argument. Afterwards {S2} said to",
# "<|endoftext|>The {PLACE} {IO} and {S1} went to had a {OBJECT}. {S2} gave it to",
# "<|endoftext|>Friends {IO} and {S1} found a {OBJECT} at the {PLACE}. {S2} gave it to",

BABA_TEMPLATES = []


def swap_substrings(s, substring_a, substring_b):
    """Swap two substrings in a string"""
    return s.replace(substring_a, "___").replace(substring_b, substring_a).replace("___", substring_b)


for template in ABBA_TEMPLATES:
    BABA_TEMPLATES.append(swap_substrings(template, "{IO}", "{S1}"))
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]
NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]


def make_arr(
    tokens: torch.Tensor,
    name: str,
    device_dtype: rc.TorchDeviceDtype = rc.TorchDeviceDtype("cuda:0", "float32"),
) -> rc.Array:
    return rc.cast_circuit(rc.Array(tokens, name=name), device_dtype.op()).cast_array()


def get_all_occurence_indices(l, x, prepend_space=True):
    """Get all occurence indices of x in l, with an optional space prepended to x."""
    if prepend_space:
        space_x = " " + x
    return [i for i, e in enumerate(l) if e == space_x or e == x]


def names_are_not_distinct(prompt):
    """
    Check that the names in the prompts are distinct.
    """
    if "IO2" in prompt:
        return prompt["IO1"] == prompt["IO2"] or prompt["IO1"] == prompt["S"] or prompt["IO2"] == prompt["S"]
    else:
        return prompt["IO"] == prompt["S"]


PromptType = Literal["mixed", "ABBA", "BABA"]


class IOIDataset:
    """Inspired by https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py,
    but not the same."""

    prompt_type: PromptType
    word_idx: dict[str, torch.Tensor]  # keys depend on the prompt family, value is tensor

    def __init__(
        self,
        N,
        prompt_type: PromptType = "mixed",
        prompt_family: Literal["IOI", "ABC"] = "IOI",
        nb_templates=None,  # if not None, limit the number of templates to use
        add_prefix_space=True,
        device="cuda:0",
        manual_metadata=None,
        seed=42,
    ):
        self.seed = seed
        random.seed(seed)
        self.N = N
        self.device = device
        self.add_prefix_space = add_prefix_space
        self.prompt_type = prompt_type
        self.prompt_family = prompt_family  # can change to "ABC" after flipping names

        if manual_metadata is not None:  # we infer the family from the metadata
            if "IO2" in manual_metadata[0].keys():
                self.prompt_family = "ABC"
            else:
                self.prompt_family = "IOI"

        if nb_templates is None:
            if prompt_type == "mixed":
                nb_templates = len(BABA_TEMPLATES) * 2
            else:
                nb_templates = len(BABA_TEMPLATES)
        if prompt_type == "ABBA":
            self.templates = ABBA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BABA":
            self.templates = BABA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "mixed":
            self.templates = (
                BABA_TEMPLATES[: (nb_templates // 2) + (nb_templates % 2)].copy()
                + ABBA_TEMPLATES[: nb_templates // 2].copy()
            )
        assert not (prompt_type == "mixed" and nb_templates % 2 != 0), "Mixed dataset with odd number of templates!"
        self.nb_templates = nb_templates

        self.tokenizer = get_interp_tokenizer()
        self.tokenizer.pad_token_id = 50256
        self.tokenizer.add_prefix_space = add_prefix_space

        self.initialize_prompts(manual_metadata=manual_metadata)
        self.initialize_word_idx()

        if self.prompt_family == "IOI":
            self.io_tokenIDs = self.prompts_toks[torch.arange(N), self.word_idx["IO"]]
            self.s_tokenIDs = self.prompts_toks[torch.arange(N), self.word_idx["S1"]]
        elif self.prompt_family == "ABC":
            self.io1_tokenIDs = self.prompts_toks[torch.arange(N), self.word_idx["IO1"]]
            self.io2_tokenIDs = self.prompts_toks[torch.arange(N), self.word_idx["IO2"]]
            self.s_tokenIDs = self.prompts_toks[torch.arange(N), self.word_idx["S"]]

    def initialize_prompts(self, manual_metadata=None):

        # define the prompts' metadata

        if manual_metadata is None:
            self.prompts_metadata = []
            for i in range(self.N):
                template_idx = random.choice(list(range(len(self.templates))))
                s = random.choice(NAMES)
                io = random.choice(NAMES)
                while io == s:
                    io = random.choice(NAMES)
                place = random.choice(PLACES)
                obj = random.choice(OBJECTS)
                self.prompts_metadata.append(
                    {
                        "S": s,
                        "IO": io,
                        "TEMPLATE_IDX": template_idx,
                        "[PLACE]": place,
                        "[OBJECT]": obj,
                        "order": "ABB" if self.templates[template_idx] in ABBA_TEMPLATES else "BAB",
                    }
                )
        else:
            self.prompts_metadata = manual_metadata

        # define the prompts' texts
        self.prompts_text = []
        for metadata in self.prompts_metadata:
            cur_template = self.templates[metadata["TEMPLATE_IDX"]]
            if self.prompt_family == "IOI":
                self.prompts_text.append(
                    cur_template.format(
                        IO=metadata["IO"],
                        S1=metadata["S"],
                        S2=metadata["S"],
                        PLACE=metadata["[PLACE]"],
                        OBJECT=metadata["[OBJECT]"],
                    )
                )
            elif self.prompt_family == "ABC":
                self.prompts_text.append(
                    cur_template.format(
                        IO=metadata["IO1"],
                        S1=metadata["IO2"],
                        S2=metadata["S"],
                        PLACE=metadata["[PLACE]"],
                        OBJECT=metadata["[OBJECT]"],
                    )
                )
            else:
                raise ValueError("Unknown prompt family")

        # define the tokens
        self.prompts_toks = torch.tensor(self.tokenizer(self.prompts_text, padding=True)["input_ids"])
        self.prompts_toks.to(self.device)

        # to get the position of the relevant names in the _tokenized_ sentences, we split the text sentences
        # by tokens, and we replace the S1, S2 IO (IO1, IO2 and S in ABC) by their annotations.

        self.prompts_text_toks = [
            [self.tokenizer.decode([x]) for x in self.tokenizer(self.prompts_text[j])["input_ids"]]
            for j in range(len(self))
        ]

        for i in range(len(self)):
            s_idx = get_all_occurence_indices(self.prompts_text_toks[i], self.prompts_metadata[i]["S"])
            if self.prompt_family == "IOI":
                io_idx = get_all_occurence_indices(self.prompts_text_toks[i], self.prompts_metadata[i]["IO"])[0]
                assert len(s_idx) == 2
                self.prompts_text_toks[i][s_idx[0]] = "{S1}"
                self.prompts_text_toks[i][s_idx[1]] = "{S2}"
                self.prompts_text_toks[i][io_idx] = "{IO}"
            elif self.prompt_family == "ABC":
                io1_idx = get_all_occurence_indices(self.prompts_text_toks[i], self.prompts_metadata[i]["IO1"])[0]
                io2_idx = get_all_occurence_indices(self.prompts_text_toks[i], self.prompts_metadata[i]["IO2"])[0]
                self.prompts_text_toks[i][io1_idx] = "{IO1}"
                self.prompts_text_toks[i][io2_idx] = "{IO2}"
                self.prompts_text_toks[i][s_idx[0]] = "{S}"

    def initialize_word_idx(self):
        self.word_idx = {}

        if self.prompt_family == "IOI":
            literals = ["{IO}", "{S1}", "{S2}"]
        elif self.prompt_family == "ABC":
            literals = ["{IO1}", "{IO2}", "{S}"]  # disjoint set of literals
        else:
            raise ValueError("Unknown prompt family")

        for word in literals:
            self.word_idx[word[1:-1]] = torch.tensor([self.prompts_text_toks[i].index(word) for i in range(len(self))])

        if self.prompt_family == "IOI":
            self.word_idx["S1+1"] = self.word_idx["S1"] + 1
        elif self.prompt_family == "ABC":
            self.word_idx["IO1+1"] = self.word_idx["IO1"] + 1  # here to be able to compare

        self.word_idx["END"] = torch.tensor([len(self.prompts_text_toks[i]) - 1 for i in range(len(self))])

    def gen_flipped_prompts(self, flip: str) -> "IOIDataset":
        """
        Return a IOIDataset where the name to flip has been replaced by a random name.
        """
        assert flip in ["IO", "S1", "S2", "IO2", "IO1", "S", "order"], "Unknown flip"
        assert (flip in ["IO", "S1", "S2", "order", "S"] and self.prompt_family == "IOI") or (
            flip in ["IO1", "IO2", "S", "order"] and self.prompt_family == "ABC"
        ), f"{flip} is illegal for prompt family {self.prompt_family}"

        new_prompts_metadata = deepcopy(self.prompts_metadata)

        if flip in ["IO", "IO1", "IO2", "S"]:  # when the flip keeps
            for prompt in new_prompts_metadata:
                prompt[flip] = random.choice(NAMES)
                while names_are_not_distinct(prompt):
                    prompt[flip] = random.choice(NAMES)
            new_family = self.prompt_family
            new_prompt_type = self.prompt_type
        elif flip == "S1":
            for prompt in new_prompts_metadata:
                prompt["IO2"] = prompt[
                    "IO"
                ]  # this lead to a change in prompt family from IOI to ABC. S stays the same.
                prompt["IO1"] = prompt["IO"]
                del prompt["IO"]
                prompt["IO2"] = random.choice(NAMES)
                while names_are_not_distinct(prompt):
                    prompt["IO2"] = random.choice(NAMES)
            new_family = "ABC"
            new_prompt_type = self.prompt_type
        elif flip == "S2":
            for prompt in new_prompts_metadata:
                prompt["IO2"] = prompt["S"]
                prompt["IO1"] = prompt["IO"]
                del prompt["IO"]
                prompt["S"] = random.choice(NAMES)
                while names_are_not_distinct(prompt):
                    prompt["S"] = random.choice(NAMES)
            new_family = "ABC"
            new_prompt_type = self.prompt_type

        elif flip == "order":
            if self.prompt_family == "IOI":
                for prompt in new_prompts_metadata:
                    prompt["TEMPLATE_IDX"] = find_flipped_template_idx(
                        prompt["TEMPLATE_IDX"], self.prompt_type, self.nb_templates
                    )

                if self.prompt_type == "ABBA":
                    new_prompt_type = "BABA"
                elif self.prompt_type == "BABA":
                    new_prompt_type = "ABBA"
                elif self.prompt_type == "mixed":
                    new_prompt_type = self.prompt_type

            if self.prompt_family == "ABC":
                new_prompt_type = self.prompt_type
                raise NotImplementedError()
                # TODO: change the order of the first two names in the prompt!

            new_family = self.prompt_type
        else:
            raise NotImplementedError()

        return IOIDataset(
            N=self.N,
            prompt_type=new_prompt_type,
            manual_metadata=new_prompts_metadata,
            prompt_family=new_family,
            nb_templates=self.nb_templates,
            add_prefix_space=self.add_prefix_space,
            device=self.device,
        )

    def __len__(self):
        return self.N


def find_flipped_template_idx(temp_idx, prompt_type, nb_templates):
    """Given a template index and the prompt type of a dataset, return the indice of the flipped template in the new dataset. This relies on the fact that the templates for the object are preserving the order from ABBA_TEMPLATES and BABA_TEMPLATES"""
    if prompt_type in ["ABBA", "BABA"]:
        return temp_idx
    elif prompt_type == "mixed":
        if temp_idx < nb_templates // 2:
            return nb_templates // 2 + temp_idx
        else:
            return temp_idx - nb_templates // 2


def add_labels_to_circuit(c: rc.Circuit, labels: torch.Tensor):
    """Run the circuit on all elements of tokens. Assumes the 'tokens' module exists in the circuit."""
    assert labels.ndim == 2 and labels.shape[1] == 2
    batch_size = labels.shape[0]
    print(batch_size)
    group = rc.DiscreteVar.uniform_probs_and_group(batch_size)
    c = c.update("labels", lambda _: rc.DiscreteVar(rc.Array(labels, name="labels"), probs_and_group=group))
    return c, group


def load_and_split_gpt2(max_len: int):
    """Only intended to be used for sudying IOI. The renaming are made to match the path patching code. See GPT2_model_loading.py for an explaination."""

    MODEL_ID = "gelu_12_tied"  # aka gpt2 small
    circ_dict, tokenizer, model_info = load_model_id(MODEL_ID)
    unbound_circuit = circ_dict["t.bind_w"]

    tokens_arr = rc.Array(torch.zeros(max_len).to(torch.long), name="tokens")
    # We use this to index into the tok_embeds to get the proper embeddings
    token_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], tokens_arr, 0, name="tok_embeds")
    bound_circuit = model_info.bind_to_input(unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"])

    transformed_circuit = bound_circuit.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
        ),
    )
    transformed_circuit = rc.conform_all_modules(transformed_circuit)

    subbed_circuit = transformed_circuit.cast_module().substitute()
    subbed_circuit = subbed_circuit.rename("logits")

    def module_but_norm(circuit: rc.Circuit):
        if isinstance(circuit, rc.Module):
            if "norm" in circuit.name or "ln" in circuit.name or "final" in circuit.name:
                return False
            else:
                return True
        return False

    for i in range(100):
        subbed_circuit = subbed_circuit.update(module_but_norm, lambda c: c.cast_module().substitute())

    renamed_circuit = subbed_circuit.update(rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner"))
    renamed_circuit = renamed_circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))

    for l in range(model_info.params.num_layers):
        # b0 -> a1.input, ... b11 -> final.input
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
        renamed_circuit = renamed_circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

        # b0.m -> m0, etc.
        renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
        renamed_circuit = renamed_circuit.update(f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias"))

        for h in range(model_info.params.num_layers):
            # b0.a.h0 -> a0.h0, etc.
            renamed_circuit = renamed_circuit.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))

    head_and_mlp_matcher = rc.IterativeMatcher(rc.Regex(r"^(a\d\d?.h\d\d?|m\d\d?)$"))
    partition = range(max_len)
    split_circuit = renamed_circuit.update(
        head_and_mlp_matcher,
        lambda c: split_to_concat(c, axis=0, partitioning_idxs=partition).rename(c.name + "_by_pos"),
    )

    new_names_dict = {}
    for l in range(model_info.params.num_layers):
        for i in range(max_len):
            for h in range(model_info.params.num_layers):
                # b0.a.h0 -> a0.h0, etc.
                new_names_dict[f"a{l}.h{h}_at_idx_{i}"] = f"a{l}_h{h}_t{i}"
            new_names_dict[f"m{l}_at_idx_{i}"] = f"m{l}_t{i}"

    split_circuit = split_circuit.update(
        rc.Matcher(*list(new_names_dict.keys())), lambda c: c.rename(new_names_dict[c.name])
    )

    return split_circuit


def load_logit_diff_model(split_circuit: rc.Circuit, io_s_labels: torch.Tensor):
    """Take GPT2 split by head and position and create a new circuit that is only computing the logit difference. The labels will be embedded in the circuit as a DiscreteVar. The function return the logit diff circuit and the group used by the DiscreteVar to sample the labels."""
    assert io_s_labels.shape[1] == 2  # a tensor of shape [nb_sentences, 2]

    device_dtype = rc.TorchDeviceDtype(dtype="float32", device="cpu")
    tokens_device_dtype = rc.TorchDeviceDtype(device_dtype.device, "int64")
    labels = make_arr(
        torch.zeros(
            2,
        ),
        "labels",
        device_dtype=tokens_device_dtype,
    )

    labels1 = rc.Index(labels, I[0], name="labels1")
    labels2 = rc.Index(labels, I[1], name="labels2")

    logit1 = rc.GeneralFunction.gen_index(
        split_circuit.index((-1,)),
        labels1,
        index_dim=0,
        batch_x=True,
        name="logit1",
    )

    logit2 = rc.GeneralFunction.gen_index(
        split_circuit.index((-1,)),
        labels2,
        index_dim=0,
        batch_x=True,
        name="logit2",
    )

    logit_diff_circuit = rc.Add.from_weighted_nodes((logit1, 1), (logit2, -1))

    logit_diff_circuit, group = add_labels_to_circuit(logit_diff_circuit, io_s_labels)
    return logit_diff_circuit, group


qkv_names = [f"a{i}.q" for i in range(12)] + [f"a{i}.k" for i in range(12)] + [f"a{i}.v" for i in range(12)]
