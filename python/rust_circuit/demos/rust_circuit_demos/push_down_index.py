# imports {{{

# %%

from __future__ import annotations

import pytest

from interp.tools.indexer import TORCH_INDEXER as I
from rust_circuit import Index, Matcher, Parser, PrintOptions, default_index_traversal, push_down_index, restrict

# %%

# }}}
# setup {{{

# %% [markdown]

# ## Setup

# %%

circuit = Parser(tensors_as_random=True, on_repeat_check_info_same=False, allow_hash_with_random=True)(
    """
0 'log_probs' [57, 50259] GeneralFunction log_softmax
  1 'logits' [57, 50259] Einsum ab,cb->ac
    2 'final.n' [57, 256] Add
      3 'final.n.w.bias' [256] Array 151e0290b45764bf65fe8e61
      4 'final.n.y_out' [57, 256] Einsum ab,a->ab
        5 'final.n.y_scale' [57, 256] Einsum ab,b->ab
          6 'final.n.z_mean' [57, 256] Einsum ab,bc->ac
            7 'final.inp' [57, 256] Add
              8 'tok_embeds' [57, 256] Array f738ed95c0f778275c1ecc91
              9 'a0' [57, 256] Einsum abc,bcd->ad
                10 'a0.comb_v' [57, 8, 32] Einsum abc,acd->bad
                  11 'a0.probs' [8, 57, 57] GeneralFunction softmax
                    12 'a0.scores' [8, 57, 57] Add
                      13 'a0.scores_mul_mask' [8, 57, 57] Einsum abc,bc->abc
                        14 'a0.scores_not_masked' [8, 57, 57] Einsum abc,adc,->abd
                          15 'a0.f_q' [8, 57, 32] Add
                            16 'a0.q' [8, 57, 32] Einsum abc,dc->adb
                              17 'a0.w.q' [8, 32, 256] Array ef8f32f62ccaa15af90ed096
                              18 'a0.n' [57, 256] Add
                                19 'a0.n.w.bias' [256] Array 925f1bb4ca52e7573018ba34
                                20 'a0.n.y_out' [57, 256] Einsum ab,a->ab
                                  21 'a0.n.y_scale' [57, 256] Einsum ab,b->ab
                                    22 'a0.n.z_mean' [57, 256] Einsum ab,bc->ac
                                      8 tok_embeds
                                      23 'a0.n.c.sub_mean' [256, 256] Array 4d11427c8d9c09763ca3c7de
                                    24 'a0.n.w.scale' [256] Array 03798fe8f36bb9e6271f8bcd
                                  25 'a0.n.full_mul' [57] GeneralFunction rsqrt
                                    26 'a0.n.c.var_p_eps' [57] Add
                                      27 'a0.n.var' [57] Einsum ab,ab,->a
                                        22 a0.n.z_mean
                                        22 a0.n.z_mean
                                        28 'a0.n.c.recip_h_size' [] Scalar 0.00390625
                                      29 'a0.n.eps' [] Scalar 0.00001
                            30 'a0.w.pos_q' [8, 57, 32] Einsum abc,dc->adb
                              17 a0.w.q
                              31 'w.pos_embeds' [57, 256] Array 47eda059261d25d4d7283df3
                          32 'a0.f_k' [8, 57, 32] Add
                            33 'a0.k' [8, 57, 32] Einsum abc,dc->adb
                              34 'a0.w.k' [8, 32, 256] Array 3eae23cb6134604792918d08
                              18 a0.n
                            35 'a0.w.pos_k' [8, 57, 32] Einsum abc,dc->adb
                              34 a0.w.k
                              31 w.pos_embeds
                          36 'a0.c.div_head_size' [] Scalar 0.17677669529663687
                        37 'a0.c.score_mask' [57, 57] Array 4d251ae1987ead35d704cd1b
                      38 'a0.c.score_neg_inf_bias' [57, 57] Array 3346b3e97cda641e9cbec006
                  39 'a0.v' [8, 57, 32] Einsum abc,dc->adb
                    40 'a0.w.v' [8, 32, 256] Array 9c803a30d26e5d50b28376dd
                    18 a0.n
                41 'a0.w.out' [8, 32, 256] Array a737a4b485f2e1cc3db22333
              42 'm0' [57, 256] Einsum ab,cb->ac
                43 'm0.act' [57, 1024] GeneralFunction gelu
                  44 'm0.add0' [57, 1024] Add
                    45 'm0.before_product0' [57, 1024] Einsum ab,cb->ac
                      46 'm0.n' [57, 256] Add
                        47 'm0.n.w.bias' [256] Array 9ae61c8299e708d1b3c12c75
                        48 'm0.n.y_out' [57, 256] Einsum ab,a->ab
                          49 'm0.n.y_scale' [57, 256] Einsum ab,b->ab
                            50 'm0.n.z_mean' [57, 256] Einsum ab,bc->ac
                              51 'm0.inp' [57, 256] Add
                                8 tok_embeds
                                9 a0
                              52 'm0.n.c.sub_mean' [256, 256] Array 4d11427c8d9c09763ca3c7de
                            53 'm0.n.w.scale' [256] Array c0b8ebacfbd81ed1c905e5d1
                          54 'm0.n.full_mul' [57] GeneralFunction rsqrt
                            55 'm0.n.c.var_p_eps' [57] Add
                              56 'm0.n.var' [57] Einsum ab,ab,->a
                                50 m0.n.z_mean
                                50 m0.n.z_mean
                                57 'm0.n.c.recip_h_size' [] Scalar 0.00390625
                              58 'm0.n.eps' [] Scalar 0.00001
                      59 'm0.w.w0' [1024, 256] Array dc59d55cbe737267d1acbbf9
                    60 'm0.w.b0' [1024] Array 7dbf23f05f2ff50ab6ae3d30
                61 'm0.w.w1' [256, 1024] Array 5002910549bc78a47941545c
              62 'm0.w.b1' [256] Array 67e1c57f81b33d2e6f6cfbe3
              63 'a1' [57, 256] Einsum abc,bcd->ad
                64 'a1.comb_v' [57, 8, 32] Einsum abc,acd->bad
                  65 'a1.probs' [8, 57, 57] GeneralFunction softmax
                    66 'a1.scores' [8, 57, 57] Add
                      67 'a1.scores_mul_mask' [8, 57, 57] Einsum abc,bc->abc
                        68 'a1.scores_not_masked' [8, 57, 57] Einsum abc,adc,->abd
                          69 'a1.f_q' [8, 57, 32] Add
                            70 'a1.q' [8, 57, 32] Einsum abc,dc->adb
                              71 'a1.w.q' [8, 32, 256] Array 91124557e0f6f11619a770d7
                              72 'a1.n' [57, 256] Add
                                73 'a1.n.w.bias' [256] Array 9c4b4d2ec8664cb710486d99
                                74 'a1.n.y_out' [57, 256] Einsum ab,a->ab
                                  75 'a1.n.y_scale' [57, 256] Einsum ab,b->ab
                                    76 'a1.n.z_mean' [57, 256] Einsum ab,bc->ac
                                      77 'a1.inp' [57, 256] Add
                                        8 tok_embeds
                                        9 a0
                                        42 m0
                                        62 m0.w.b1
                                      78 'a1.n.c.sub_mean' [256, 256] Array 4d11427c8d9c09763ca3c7de
                                    79 'a1.n.w.scale' [256] Array 0fcb1f000bb89dda2e86c04f
                                  80 'a1.n.full_mul' [57] GeneralFunction rsqrt
                                    81 'a1.n.c.var_p_eps' [57] Add
                                      82 'a1.n.var' [57] Einsum ab,ab,->a
                                        76 a1.n.z_mean
                                        76 a1.n.z_mean
                                        83 'a1.n.c.recip_h_size' [] Scalar 0.00390625
                                      84 'a1.n.eps' [] Scalar 0.00001
                            85 'a1.w.pos_q' [8, 57, 32] Einsum abc,dc->adb
                              71 a1.w.q
                              31 w.pos_embeds
                          86 'a1.f_k' [8, 57, 32] Add
                            87 'a1.k' [8, 57, 32] Einsum abc,dc->adb
                              88 'a1.w.k' [8, 32, 256] Array 15f4e9cd023913a9a854a514
                              72 a1.n
                            89 'a1.w.pos_k' [8, 57, 32] Einsum abc,dc->adb
                              88 a1.w.k
                              31 w.pos_embeds
                          90 'a1.c.div_head_size' [] Scalar 0.17677669529663687
                        91 'a1.c.score_mask' [57, 57] Array 4d251ae1987ead35d704cd1b
                      92 'a1.c.score_neg_inf_bias' [57, 57] Array 3346b3e97cda641e9cbec006
                  93 'a1.v' [8, 57, 32] Einsum abc,dc->adb
                    94 'a1.w.v' [8, 32, 256] Array 5aaef038d2194d529aea8758
                    72 a1.n
                95 'a1.w.out' [8, 32, 256] Array 4bfcebee9a186581d8de84cc
              96 'm1' [57, 256] Einsum ab,cb->ac
                97 'm1.act' [57, 1024] GeneralFunction gelu
                  98 'm1.add0' [57, 1024] Add
                    99 'm1.before_product0' [57, 1024] Einsum ab,cb->ac
                      100 'm1.n' [57, 256] Add
                        101 'm1.n.w.bias' [256] Array c8ee8c692edc6d92e7c2c219
                        102 'm1.n.y_out' [57, 256] Einsum ab,a->ab
                          103 'm1.n.y_scale' [57, 256] Einsum ab,b->ab
                            104 'm1.n.z_mean' [57, 256] Einsum ab,bc->ac
                              105 'm1.inp' [57, 256] Add
                                8 tok_embeds
                                9 a0
                                42 m0
                                62 m0.w.b1
                                63 a1
                              106 'm1.n.c.sub_mean' [256, 256] Array 4d11427c8d9c09763ca3c7de
                            107 'm1.n.w.scale' [256] Array 617a80d2cb0cb45318ef9885
                          108 'm1.n.full_mul' [57] GeneralFunction rsqrt
                            109 'm1.n.c.var_p_eps' [57] Add
                              110 'm1.n.var' [57] Einsum ab,ab,->a
                                104 m1.n.z_mean
                                104 m1.n.z_mean
                                111 'm1.n.c.recip_h_size' [] Scalar 0.00390625
                              112 'm1.n.eps' [] Scalar 0.00001
                      113 'm1.w.w0' [1024, 256] Array c46ef0b6b06fa0a25a9e8f74
                    114 'm1.w.b0' [1024] Array 1eeee3393a8ef176b116cec0
                115 'm1.w.w1' [256, 1024] Array 040df31a221c659c1da228b7
              116 'm1.w.b1' [256] Array 7010bd45175fc579786a025a
            117 'final.n.c.sub_mean' [256, 256] Array 4d11427c8d9c09763ca3c7de
          118 'final.n.w.scale' [256] Array 3900ff7658d3f069ae29926e
        119 'final.n.full_mul' [57] GeneralFunction rsqrt
          120 'final.n.c.var_p_eps' [57] Add
            121 'final.n.var' [57] Einsum ab,ab,->a
              6 final.n.z_mean
              6 final.n.z_mean
              122 'final.n.c.recip_h_size' [] Scalar 0.00390625
            123 'final.n.eps' [] Scalar 0.00001
    124 'w.unembed' [50259, 256] Array 460a894d3c021fb94cabdfb1
"""
)

# %%
# }}}
# %%

pushed = push_down_index(
    Index(circuit, [17]), restrict(default_index_traversal(), term_early_at="final.inp"), suffix="_idx"
)
pushed.print(PrintOptions(shape_only_when_necessary=False))

# %%

pushed_many = push_down_index(
    Index(circuit, I[2:7]), restrict(default_index_traversal(), term_early_at="final.inp"), suffix="_idx"
)
pushed_many.print(PrintOptions(shape_only_when_necessary=False))

# %%

pushed_sub = push_down_index(Index(pushed_many, I[1:3]), default_index_traversal(), suffix="_IDX")
pushed_sub.print(PrintOptions(shape_only_when_necessary=False))

# %%

pushed_far = push_down_index(Index(circuit, I[2:7]), suffix="_idx")  # push down all the way!
pushed_far.print(PrintOptions(shape_only_when_necessary=False))

_ = Matcher("m1_idx").get_unique(pushed_far)
with pytest.raises(RuntimeError):
    Matcher("m1").get_unique(pushed_far)  # this doesn't exist
_ = Matcher("a1_idx").get_unique(pushed_far)
with pytest.raises(RuntimeError):
    Matcher("a1").get_unique(pushed_far)  # this doesn't exist
# but both of these will exist due to sequence position translation from a1
_ = Matcher("m0_idx").get_unique(pushed_far)
_ = Matcher("m0").get_unique(pushed_far)
_ = Matcher("a0_idx").get_unique(pushed_far)
_ = Matcher("a0").get_unique(pushed_far)
