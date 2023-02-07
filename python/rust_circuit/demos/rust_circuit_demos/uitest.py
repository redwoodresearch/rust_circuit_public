import rust_circuit as rc
from rust_circuit import Parser
from rust_circuit.ui.ui import circuit_graph_ui

parent = Parser()(
    """0 Add
  4 'asdasd' Add
    1 'asdf' Add
      2 [1] Add
        6 'Z' [1] Scalar 12
  5 'ashudashk' Add
    3 'whe' Add
      2"""
)
parent = Parser()(
    """'b' Add # line length: 4 # node count: 69
  'b.m.set' [2s,0s] SetSymbolicShape # line length: 25 # node count: 68
    'b.m' Module # line length: 7 # node count: 67
      'm.ln_call' Module # line length: 7 # node count: 28
        'm' Einsum h,oh->o # line length: 15 # node count: 8
          'm.act' GeneralFunction gelu # line length: 21 # node count: 6
            'm.pre' Add # line length: 4 # node count: 5
              'm.pre_mul' Einsum i,hi->h # line length: 15 # node count: 3
                'm.input' [0s] Symbol 13ec4cdc-5f25-4969-870b-0bfa2300187b # line length: 49 # node count: 1
                'm.w.proj_in' [6s,0s] Symbol 5217f963-0cdb-460e-bb1f-f82f7fbb3cd9 # line length: 52 # node count: 1
              'm.w.in_bias' [6s] Symbol c870ec00-8c6f-4080-907c-703ea85dde48 # line length: 49 # node count: 1
          'm.w.proj_out' [1s,6s] Symbol fdefa9af-a7d6-4a38-a7ed-5ce816c6efe7 # line length: 52 # node count: 1
        'm.ln' Module ! 'm.input' # line length: 7 # node count: 19
          'ln' Add # line length: 4 # node count: 15
            'ln.w.bias' [0s] Symbol 621c7792-0177-45ab-87c5-7ff1c3bec487 # line length: 49 # node count: 1
            'ln.y_scaled' Einsum h,h->h # line length: 14 # node count: 13
              'ln.y' Einsum h,->h # line length: 13 # node count: 11
                'ln.mean_subbed' Add # line length: 4 # node count: 6
                  'ln.input' [0s] Symbol 981b4d2a-711b-4a9d-a11c-d859c311e80c # line length: 49 # node count: 1
                  'ln.neg_mean' Einsum h,z,->z # line length: 15 # node count: 5
                    'ln.input' # line length: 49 # node count: 1
                    'ln.neg' [1] Scalar -1 # line length: 14 # node count: 1
                    'ln.c.recip_hidden_size' GeneralFunction reciprocal # line length: 27 # node count: 3
                      'ln.c.hidden_size' GeneralFunction last_dim_size # line length: 30 # node count: 2
                        'ln.input' # line length: 49 # node count: 1
                'ln.rsqrt' GeneralFunction rsqrt # line length: 22 # node count: 10
                  'ln.var_p_eps' Add # line length: 4 # node count: 9
                    'ln.c.eps' [] Scalar 0.00001 # line length: 18 # node count: 1
                    'ln.var' Einsum h,h,-> # line length: 14 # node count: 7
                      'ln.mean_subbed' # line length: 4 # node count: 6
                      'ln.mean_subbed' # line length: 4 # node count: 6
                      'ln.c.recip_hidden_size' # line length: 27 # node count: 3
              'ln.w.scale' [0s] Symbol 0fa341c3-34b3-4699-847f-08674808b28a # line length: 49 # node count: 1
          'm.ln.input' [2s,0s] Symbol a0fe2ee1-77bc-4afd-bf98-3b34212d944b ! 'ln.input' # line length: 52 # node count: 1
          'm.ln.w.bias' [0s] Symbol 049d1137-70a9-495b-b0d1-01625ec05540 ! 'ln.w.bias' # line length: 49 # node count: 1
          'm.ln.w.scale' [0s] Symbol b325c75f-f8b8-41c5-81d9-4ed21dc82208 ! 'ln.w.scale' # line length: 49 # node count: 1
      'b.m.input' Add ! 'm.ln.input' # line length: 4 # node count: 53
        'b.a.set' [2s,0s] SetSymbolicShape # line length: 25 # node count: 52
          'b.a' Module # line length: 7 # node count: 51
            'a.ln_call' Module # line length: 7 # node count: 49
              'a.on_inp' Module # line length: 7 # node count: 29
                'a' Einsum shV,hdV->sd # line length: 19 # node count: 27
                  'a.comb_v' Einsum hqk,hkV->qhV # line length: 20 # node count: 25
                    'a.attn_probs' GeneralFunction softmax # line length: 24 # node count: 21
                      'a.attn_scores' Add # line length: 4 # node count: 20
                        'a.attn_scores_raw' Einsum hqc,hkc,,qk->hqk # line length: 24 # node count: 11
                          'a.q' Einsum qd,hcd->hqc # line length: 19 # node count: 3
                            'a.q.input' [3s,0s] Symbol 4f80d1a1-86a4-4e44-94f7-909ec7089061 # line length: 52 # node count: 1
                            'a.w.q' [5s,8s,0s] Symbol 665efa60-d86c-40d5-92b2-b96d11686a8b # line length: 55 # node count: 1
                          'a.k' Einsum kd,hcd->hkc # line length: 19 # node count: 3
                            'a.k.input' [4s,0s] Symbol 664bddee-28ca-47e7-9fb7-9a718de06619 # line length: 52 # node count: 1
                            'a.w.k' [5s,8s,0s] Symbol 41177709-446d-4588-b9e5-c2bbf59d53a0 # line length: 55 # node count: 1
                          'a.c.div_head_size' GeneralFunction rsqrt # line length: 22 # node count: 4
                            'a.c.head_size' GeneralFunction last_dim_size # line length: 30 # node count: 3
                              'a.c.bias_for_head_size' Einsum ijk->j # line length: 14 # node count: 2
                                'a.w.k' # line length: 55 # node count: 1
                          'a.mask' [3s,4s] Symbol ccfe5bc9-b402-42dd-a5e1-191e6fb7c268 # line length: 52 # node count: 1
                        'a.score_neg_inf_bias' Einsum qk,o->oqk # line length: 17 # node count: 9
                          'a.not_mask' Module # line length: 7 # node count: 7
                            'not_mask' Add # line length: 4 # node count: 5
                              'one' [] Scalar 1 # line length: 12 # node count: 1
                              'not_mask.neg_mask' Einsum ,-> # line length: 11 # node count: 3
                                'not_mask.input' [] Symbol b46f6370-11e1-4535-aabc-94554c234673 # line length: 47 # node count: 1
                                'neg_one' [] Scalar -1 # line length: 13 # node count: 1
                            'a.mask' ! 'not_mask.input' # line length: 52 # node count: 1
                          'a.neg_inf' [1] Scalar -10000 # line length: 18 # node count: 1
                    'a.v' Einsum kd,hVd->hkV # line length: 19 # node count: 3
                      'a.v.input' [4s,0s] Symbol 8fd4c632-7f28-49ee-84cc-3dde997e0693 # line length: 52 # node count: 1
                      'a.w.v' [5s,9s,0s] Symbol 79b6ebff-f9d0-411a-bcdc-530cc13e1524 # line length: 55 # node count: 1
                  'a.w.o' [5s,1s,9s] Symbol 11a116cb-2168-4725-a06f-1b61a8ca6797 # line length: 55 # node count: 1
                'a.input' [2s,0s] Symbol f9eabd07-e2ab-4ed4-8b4a-c9c039d61835 ! 'a.q.input' # line length: 52 # node count: 1
                'a.input' ! 'a.k.input' # line length: 52 # node count: 1
                'a.input' ! 'a.v.input' # line length: 52 # node count: 1
              'a.ln' Module ! 'a.input' # line length: 7 # node count: 19
                'ln' # line length: 4 # node count: 15
                'a.ln.input' [2s,0s] Symbol 85fb9501-ce13-4aa5-ab98-558020a3daec ! 'ln.input' # line length: 52 # node count: 1
                'a.ln.w.bias' [0s] Symbol cecc5585-abab-461e-afcd-664a1dd80037 ! 'ln.w.bias' # line length: 49 # node count: 1
                'a.ln.w.scale' [0s] Symbol e8f6c84d-9593-4383-8f5d-76b2ea589c8e ! 'ln.w.scale' # line length: 49 # node count: 1
            'b.input' [2s,0s] Symbol 5837c4fd-f5ac-4bff-8456-abf3e95bcf36 ! 'a.ln.input' # line length: 52 # node count: 1
        'b.input' # line length: 52 # node count: 1
  'b.m.input' # line length: 4 # node count: 53"""
)

circuit_graph_ui(
    parent,
    annotators={"test": lambda circ: circ.name, "children": lambda circ: str(circ.num_children)},
    enabled_annotators=["children"],
    # default_hidden=parent.get(ui_default_hidden_matcher),
    default_shown=parent.get(rc.Matcher(rc.Einsum, rc.Add, rc.Module)),
)
