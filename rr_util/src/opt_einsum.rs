use std::{
    cmp::{max, min},
    iter::zip,
    rc::Rc,
    sync::Mutex,
};

use anyhow::{bail, Result};
use num_bigint::BigUint;
use once_cell::sync::Lazy;
/// This file optimizes tensor contraction order, like Python's opt_einsum
/// It specifies contraction order using rotating id's, like opt_einsum or numpy,
/// which is kindof awkward but compatible
/// worth noting our tensor contractions are hypergraphs or "non standard" in tensor contraction literature

/// current algorithm is:
/// 1: contract tensors where one's indices are a subset of the other's
/// 2: find disconnected subgraphs
/// 3: for each subgraph, optimize using DP algorithm based on opt_einsum's DP
/// 4: contract subgraphs together
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    name::Name,
    tensor_util::ParseError,
    union_find::UnionFind,
    util::{dict_to_list, EinsumAxes, NamedAxes, ALPHABET, ALPHABET_INV},
};

/// returns vec of vecs of indices of operands
pub fn get_disconnected_sugraphs(
    operand_ints: &Vec<Vec<usize>>,
    ints_to_operands: &HashMap<usize, Vec<usize>>,
) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(operand_ints.len());
    for (_int, operands) in ints_to_operands.iter() {
        if operands.len() >= 2 {
            let (first, rest) = operands.split_first().unwrap();
            for aftery in rest {
                uf.union(*first, *aftery);
            }
        }
    }
    uf.to_vec_vec()
}

pub fn get_int_to_tensor_appearance(tensor_ints: &[Vec<usize>]) -> HashMap<usize, Vec<usize>> {
    let mut int_to_tensor_appearance: HashMap<usize, Vec<usize>> = HashMap::default();
    for (i, tensor_ints) in tensor_ints.iter().enumerate() {
        for inty in tensor_ints.iter() {
            int_to_tensor_appearance
                .entry(*inty)
                .or_insert(Vec::new())
                .push(i);
        }
    }
    int_to_tensor_appearance
}

pub fn difference(yes: usize, no: usize, all: usize) -> usize {
    yes & (all ^ no)
}

fn ints_to_bitmap(ints: &Vec<usize>) -> usize {
    let mut result: usize = 0;
    for inty in ints.iter() {
        result |= 1 << inty;
    }
    result
}

fn bitmask_to_ints(bitmask: usize) -> Vec<usize> {
    let mut result = vec![];
    let mut cur_bitmap = bitmask;
    loop {
        let trailing_zeros = cur_bitmap.trailing_zeros() as usize;
        if trailing_zeros < 64 {
            cur_bitmap ^= 1 << trailing_zeros;
            result.push(trailing_zeros);
        } else {
            break;
        }
    }
    result
}

fn bitmap_and_sizes_to_size(bitmap: usize, int_sizes: &[usize]) -> usize {
    let mut result: usize = 1;
    let mut cur_bitmap = bitmap;
    loop {
        let trailing_zeros = cur_bitmap.trailing_zeros() as usize;
        if trailing_zeros < int_sizes.len() {
            cur_bitmap ^= 1 << trailing_zeros;
            result = result.saturating_mul(int_sizes[trailing_zeros]);
        } else {
            break;
        }
    }
    result
}

fn is_outer(a: usize, b: usize) -> bool {
    a & b == 0
}

fn get_outer_indices(
    operands: usize,
    indices: usize,
    all_operands_bmp: usize,
    tensor_ints: &[usize],
    outer_ints: usize,
) -> usize {
    let complement = all_operands_bmp ^ operands;
    let mut result: usize = outer_ints & indices;
    let mut cur_bitmap = complement;
    loop {
        let trailing_zeros = cur_bitmap.trailing_zeros() as usize;
        if trailing_zeros < tensor_ints.len() {
            cur_bitmap ^= 1 << trailing_zeros;
            result |= tensor_ints[trailing_zeros] & indices;
        } else {
            break;
        }
    }
    result
}

fn independent_tree_contraction(
    dpes: &Vec<DpEntry>,
    tensor_ints: &[usize],
    all_operands_bmp: usize,
    outer_ints: usize,
    int_sizes: &[usize],
) -> DpEntry {
    // if they're independent, it's just outer product, and we go greedily on size
    let mut dps: Vec<DpEntry> = dpes.clone();
    while dps.len() > 1 {
        dps.sort_by_key(|x| (usize::MAX - bitmap_and_sizes_to_size(x.indices, int_sizes)));
        let to_contract = (dps.pop().unwrap(), dps.pop().unwrap());
        let new_indices = to_contract.0.indices | to_contract.1.indices;
        let new_entry = DpEntry {
            indices: get_outer_indices(
                new_indices,
                new_indices,
                all_operands_bmp,
                tensor_ints,
                outer_ints,
            ), // BUG include indices
            operands: 0,
            cost: 0,
            contraction: Rc::new(Contraction::Composed((
                Rc::new(to_contract.0),
                Rc::new(to_contract.1),
            ))),
        };
        dps.push(new_entry)
    }
    dps[0].clone()
}

#[derive(Debug, Clone, Hash)]
enum Contraction {
    Operand(usize),
    Composed((Rc<DpEntry>, Rc<DpEntry>)),
}
#[derive(Debug, Clone, Hash)]
struct DpEntry {
    indices: usize,
    operands: usize,
    cost: usize,
    contraction: Rc<Contraction>,
}

fn contraction_to_ssa_ids(contraction: &DpEntry, num_tensors: usize) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    fn recurse(result: &mut Vec<Vec<usize>>, nt: usize, c: &DpEntry) -> usize {
        match (*c.contraction).clone() {
            Contraction::Operand(x) => x,
            Contraction::Composed(x) => {
                let recursed = (recurse(result, nt, &x.0), recurse(result, nt, &x.1));
                result.push(vec![recursed.0, recursed.1]);
                nt + result.len() - 1
            }
        }
    }
    recurse(&mut result, num_tensors, contraction);
    result
}

fn contraction_ssa_ids_to_recycled(
    contractions: &Vec<Vec<usize>>,
    num_tensors: usize,
) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut map: Vec<i64> = (0..(num_tensors + contractions.len() + 1) as i64).collect();
    for c in contractions.iter() {
        result.push(c.iter().map(|x| map[*x] as usize).collect());
        for i in c.iter() {
            for j in *i..map.len() {
                map[j] -= 1;
            }
        }
    }
    result
}

#[pyclass]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EinsumSpec {
    #[pyo3(get)]
    pub input_ints: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub output_ints: Vec<usize>,
    #[pyo3(get)]
    pub int_sizes: Vec<usize>,
}
#[pymethods]
impl EinsumSpec {
    #[new]
    fn new(
        operand_ints: Vec<Vec<usize>>,
        out_ints: Vec<usize>,
        int_sizes: &PyDict,
    ) -> Result<Self> {
        let sizes_dict: HashMap<usize, usize> = int_sizes.extract()?;
        let int_sizes = dict_to_list(&sizes_dict, None);
        let result = EinsumSpec {
            input_ints: operand_ints,
            output_ints: out_ints,
            int_sizes,
        };
        if result.validate() {
            return Ok(result);
        }
        bail!("invalid")
    }
    pub fn ints_in_input(&self) -> HashSet<usize> {
        let ints_in_input: HashSet<usize> = self.input_ints.iter().flatten().copied().collect();
        ints_in_input
    }
    pub fn validate(&self) -> bool {
        let ints_in_input = self.ints_in_input();
        let all_output_in_input = self.output_ints.iter().all(|x| ints_in_input.contains(x));
        let all_ints_have_sizes = ints_in_input
            .iter()
            .all(|x| (*x < self.int_sizes.len()) && self.int_sizes[*x] > 0);
        all_output_in_input && all_ints_have_sizes
    }
    pub fn shapes(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        (
            self.input_ints
                .iter()
                .map(|operand_ints| {
                    operand_ints
                        .iter()
                        .map(|inty| self.int_sizes[*inty])
                        .collect()
                })
                .collect(),
            self.output_ints
                .iter()
                .map(|inty| self.int_sizes[*inty])
                .collect(),
        )
    }

    pub fn normalize(&self) -> EinsumSpec {
        let mut ints_map: HashMap<usize, usize> = HashMap::default();
        for operand_ints in &self.input_ints {
            for o_int in operand_ints {
                let z = ints_map.len(); // take length before we borrow as mutable? okay
                ints_map.entry(*o_int).or_insert(z);
            }
        }
        self.int_map(&ints_map)
    }

    pub fn to_einsum_string(&self) -> String {
        let operand_letters = self
            .input_ints
            .iter()
            .map(|one_operand_ints| {
                one_operand_ints
                    .iter()
                    .map(|i| ALPHABET[*i].clone())
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join(",");
        let out_letters = self
            .output_ints
            .iter()
            .map(|i| ALPHABET[*i].clone())
            .collect::<Vec<String>>()
            .join("");
        operand_letters + "->" + &out_letters
    }

    /// Use self and children's named axes for axes names. Return None if not enough named axes.
    /// If different named are given to the same axes, use the axes given by self (highest priority), or the first child, or the second, ...
    pub fn to_fancy_einsum_string(
        &self,
        self_named_axes: NamedAxes,
        children_named_axes: Vec<NamedAxes>,
    ) -> Option<String> {
        let mut axis_to_name: HashMap<usize, Name> = Default::default();

        // self named axes have priority
        for (i, name) in self_named_axes.iter() {
            axis_to_name.insert(self.output_ints[*i as usize], *name);
        }

        let operand_words = zip(children_named_axes.iter(), self.input_ints.iter())
            .map(|(named_axes, one_operand_ints)| {
                if named_axes.len() < one_operand_ints.len() {
                    return None; // No hope of having a name for each axis
                }
                Some(
                    one_operand_ints
                        .iter()
                        .enumerate()
                        .map(|(i, a)| {
                            if let Some(o_name) = axis_to_name.get(a) {
                                Some((*o_name).into())
                            } else {
                                let name = named_axes.get(&(i as u8))?;
                                axis_to_name.insert(*a, *name);
                                Some((*name).into())
                            }
                        })
                        .collect::<Option<Vec<String>>>()?
                        .join(" "),
                )
            })
            .collect::<Option<Vec<String>>>()?
            .join(", ");
        if axis_to_name.len() > HashSet::from_iter(axis_to_name.values().cloned()).len() {
            return None; // Can't tolerate duplicate names
        }
        let out_words = self
            .output_ints
            .iter()
            .map(|a| (axis_to_name.get(a).unwrap().string().to_owned())) // output ints included in input ints
            .collect::<Vec<String>>()
            .join(" ");
        Some(format!("fancy: {operand_words} -> {out_words}"))
    }

    pub fn to_maybe_fancy_string(
        &self,
        self_named_axes: NamedAxes,
        children_named_axes: Vec<NamedAxes>,
    ) -> String {
        if self_named_axes.is_empty() {
            self.to_einsum_string() // in this case we override and don't use fancy which
                                    // just adds extra stuff we don't want
        } else if let Some(fancy_string) =
            self.to_fancy_einsum_string(self_named_axes, children_named_axes)
        {
            fancy_string
        } else {
            self.to_einsum_string()
        }
    }

    pub fn flops(&self) -> BigUint {
        if self.input_ints.is_empty() {
            return BigUint::from(0usize);
        }
        let used_ints = self.ints_in_input();
        let used_int_sizes = used_ints.iter().map(|x| self.int_sizes[*x]);
        let used_int_sizes_prod: BigUint = used_int_sizes
            .map(BigUint::from)
            .fold(BigUint::from(1usize), |a, b| a * b);
        let muls: BigUint = BigUint::from(self.input_ints.len() - 1) * used_int_sizes_prod;
        let output_size: BigUint = self
            .output_ints
            .iter()
            .map(|x| BigUint::from(self.int_sizes[*x]))
            .fold(BigUint::from(1usize), |a, b| a * b);

        let ints_in_output = HashSet::from_iter(self.output_ints.iter().copied());
        let non_output_size: BigUint = used_ints
            .difference(&ints_in_output)
            .map(|x| BigUint::from(self.int_sizes[*x]))
            .fold(BigUint::from(1usize), |a, b| a * b);
        let sums = output_size * (non_output_size - 1usize);
        muls + sums
    }

    #[staticmethod]
    pub fn string_to_ints(string: String) -> Result<(Vec<EinsumAxes>, EinsumAxes)> {
        if let [inputs, output] = string.split("->").collect::<Vec<&str>>()[..] {
            let output_ints = output
                .chars()
                .filter(|x| x != &' ')
                .map(|x| {
                    let out = ALPHABET_INV.get(&x.to_string()).cloned().ok_or(
                        ParseError::EinsumStringInvalid {
                            string: string.clone(),
                            substring: x.to_string(),
                        },
                    )? as u8;
                    Ok(out)
                })
                .collect::<Result<_>>()?;
            let input_ints = inputs
                .split(',')
                .map(|string| {
                    string
                        .chars()
                        .filter(|x| x != &' ')
                        .map(|c| {
                            let out = ALPHABET_INV.get(&c.to_string()).cloned().ok_or(
                                ParseError::EinsumStringInvalid {
                                    string: string.to_owned(),
                                    substring: c.to_string(),
                                },
                            )? as u8;
                            Ok(out)
                        })
                        .collect::<Result<_>>()
                })
                .collect::<Result<_>>()?;
            Ok((input_ints, output_ints))
        } else {
            bail!(ParseError::EinsumStringNoArrow {
                string: string.to_owned(),
            })
        }
    }

    #[staticmethod]
    pub fn fancy_string_to_ints(string: String) -> Result<(Vec<EinsumAxes>, EinsumAxes)> {
        if let [inputs, output] = string.split("->").collect::<Vec<&str>>()[..] {
            let mut string_to_int: HashMap<&str, u8> = HashMap::default();
            let input_ints = inputs
                .split(',')
                .map(|string| {
                    string
                        .split_whitespace()
                        .map(|x| {
                            let l = string_to_int.len() as u8;
                            *string_to_int.entry(x).or_insert(l)
                        })
                        .collect()
                })
                .collect();
            let output_ints = output
                .split_whitespace()
                .map(|x| {
                    let l = string_to_int.len() as u8;
                    *string_to_int.entry(x).or_insert(l)
                })
                .collect();
            Ok((input_ints, output_ints))
        } else {
            bail!(ParseError::EinsumStringNoArrow {
                string: string.to_owned(),
            })
        }
    }

    pub fn optimize_pre_contract_subsets(
        &self,
        dont_contract_subsets_into: Option<Vec<usize>>,
    ) -> (Vec<Vec<usize>>, Vec<usize>) {
        let dont = dont_contract_subsets_into.unwrap_or_default();
        let mut result: Vec<Vec<usize>> = vec![];
        let mut stack: Vec<usize> = self.input_ints.iter().map(ints_to_bitmap).collect();
        loop {
            let mut res = vec![0, 0];
            let mut big_sup: usize = 0;
            let mut candidate_cost: usize = usize::MAX;
            for (i, sub) in stack.iter().enumerate() {
                for (j, sup) in stack.iter().enumerate() {
                    if !dont.contains(&j)
                        && i != j
                        && sup | sub == *sup
                        && bitmap_and_sizes_to_size(*sup, &self.int_sizes) <= candidate_cost
                    {
                        res[0] = min(i, j);
                        res[1] = max(i, j);
                        big_sup = *sup;
                        candidate_cost = bitmap_and_sizes_to_size(*sup, &self.int_sizes);
                    }
                }
            }
            if res[..] != [0, 0] {
                stack.remove(res[1]);
                stack.remove(res[0]);
                stack.push(big_sup);
                result.push(res);
            } else {
                break;
            }
        }
        result.push((0..stack.len()).collect());

        (result, stack)
    }

    /// based on opt_einsum's dynamic programming algorithm https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/paths.py#L1124
    /// with one major added feature to bound runtime in the worst case: a cap on the number of held contractions with the same number of tensors
    /// so that we never have to fallback to another algorithm
    pub fn optimize_dp(
        &self,
        check_outer: Option<bool>,
        mem_cap: Option<usize>,
        hash_map_cap: Option<usize>,
    ) -> Result<Vec<Vec<usize>>> {
        let hash_map_cap = hash_map_cap.unwrap_or(70);
        let mem_cap = mem_cap.unwrap_or(usize::MAX);
        if self.input_ints.len() == 1 {
            return Ok(vec![vec![0]]);
        }

        let tensor_ints: Vec<Vec<usize>> = self.input_ints.clone();
        let subgraphs =
            get_disconnected_sugraphs(&tensor_ints, &get_int_to_tensor_appearance(&tensor_ints));
        let check_outer = check_outer.unwrap_or(false);
        let use_blacklist = false;
        let all_operands_bmp: usize = ints_to_bitmap(&(0..tensor_ints.len()).collect());
        let tensor_bmps_used: Vec<usize> = tensor_ints.iter().map(ints_to_bitmap).collect();
        let out_bmp = ints_to_bitmap(&self.output_ints);

        let mut subgraph_results: Vec<Rc<DpEntry>> = Vec::new();
        for sg in subgraphs.iter() {
            let sg_bmp: usize = ints_to_bitmap(sg);
            let mut cost_cap: usize = max(
                2,
                bitmap_and_sizes_to_size(sg_bmp & out_bmp, &self.int_sizes) * 64,
            );
            let cost_cap_increment: usize = 8;
            assert!(
                sg.len() < 64,
                "store bitmaps as usize, so can only have 64 per subgraph"
            );
            // unsafe {breakpoint()}
            let mut dp: Vec<HashMap<usize, Rc<DpEntry>>> =
                (0..sg.len() + 1).map(|_x| HashMap::default()).collect();
            let mut dominated_sets: HashSet<usize> = HashSet::default();
            let mut dominated_sets_here: HashSet<usize> = HashSet::default();
            dp[1] = sg
                .iter()
                .map(|i| {
                    (
                        1 << i,
                        Rc::new(DpEntry {
                            indices: ints_to_bitmap(&tensor_ints[*i]),
                            operands: 1 << i,
                            cost: 0,
                            contraction: Rc::new(Contraction::Operand(*i)),
                        }),
                    )
                })
                .into_iter()
                .collect();
            while dp.last().unwrap().is_empty() {
                for n in 2..dp[1].len() + 1 {
                    // need this bc iter earier idxs while updating later
                    let (dp_prev, dp_here_vec) = dp.split_at_mut(n);
                    for m in 1..(n / 2 + 1) {
                        for (s1, dp_entry_1) in dp_prev[m].iter() {
                            for (s2, dp_entry_2) in dp_prev[n - m].iter() {
                                if s1 & s2 == 0
                                    && (check_outer
                                        || !is_outer(dp_entry_1.indices, dp_entry_2.indices))
                                {
                                    let union = dp_entry_1.indices | dp_entry_2.indices;
                                    let new_cost = dp_entry_1
                                        .cost
                                        .checked_add(dp_entry_2.cost)
                                        .unwrap_or(usize::MAX)
                                        .checked_add(bitmap_and_sizes_to_size(
                                            union,
                                            &self.int_sizes,
                                        ))
                                        .unwrap_or(usize::MAX);
                                    let new_operands = s1 | s2;
                                    let op_set_already = dp_here_vec[0].contains_key(&new_operands);
                                    if new_cost < cost_cap
                                        && (!op_set_already
                                            || new_cost < dp_here_vec[0][&new_operands].cost)
                                    {
                                        if op_set_already && use_blacklist {
                                            match &*(dp_here_vec[0][&new_operands]
                                                .contraction
                                                .clone())
                                            {
                                                Contraction::Composed(t) => {
                                                    let left = &t.0;
                                                    let right = &t.1;
                                                    if left.cost > new_cost {
                                                        dominated_sets.insert(left.operands);
                                                        dominated_sets_here.insert(left.operands);
                                                    }
                                                    if right.cost > new_cost {
                                                        dominated_sets.insert(right.operands);
                                                        dominated_sets_here.insert(right.operands);
                                                    }
                                                }
                                                Contraction::Operand(_) => {}
                                            }
                                        }
                                        let new_indices = get_outer_indices(
                                            new_operands,
                                            union,
                                            sg_bmp,
                                            &tensor_bmps_used,
                                            out_bmp,
                                        );
                                        if bitmap_and_sizes_to_size(new_indices, &self.int_sizes)
                                            < mem_cap
                                            && (op_set_already
                                                || dp_here_vec[0].len() < hash_map_cap)
                                        {
                                            dp_here_vec[0].insert(
                                                new_operands,
                                                Rc::new(DpEntry {
                                                    indices: new_indices,
                                                    operands: new_operands,
                                                    cost: new_cost,
                                                    contraction: Rc::new(Contraction::Composed((
                                                        dp_entry_1.clone(),
                                                        dp_entry_2.clone(),
                                                    ))),
                                                }),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for d in dominated_sets_here.iter() {
                        dp[d.count_ones() as usize].remove(d);
                    }
                    dominated_sets_here.clear();
                }
                let new_cost_checked = cost_cap.checked_mul(cost_cap_increment);
                match new_cost_checked {
                    None => bail!("einsum opt couldn't find solution within cost cap"),
                    Some(x) => cost_cap = x,
                }
            }

            subgraph_results.push(dp.last().unwrap()[&sg_bmp].clone());
        }
        let overall_contraction = independent_tree_contraction(
            &subgraph_results.iter().map(|x| (**x).clone()).collect(), /* maybe should use rc_unwrap_or_clone */
            &tensor_bmps_used,
            all_operands_bmp,
            out_bmp,
            &self.int_sizes,
        );
        // unsafe{breakpoint()}
        let ssa_result = contraction_to_ssa_ids(&overall_contraction, tensor_ints.len());
        assert!(
            &ssa_result.iter().all(|x| x.len() == 2),
            "{:?}",
            &ssa_result
        );
        let result = contraction_ssa_ids_to_recycled(&ssa_result, tensor_ints.len());
        assert!(&result.iter().all(|x| x.len() == 2), "{:?}", &result);
        Ok(result)
    }

    pub fn optimize(
        &self,
        check_outer: Option<bool>,
        mem_cap: Option<usize>,
        hash_map_cap: Option<usize>,
        dont_contract_subsets_into: Option<Vec<usize>>,
    ) -> Result<Vec<Vec<usize>>> {
        if let Some(mx) = self.input_ints.iter().flatten().max() && mx > &63{
            bail!("rust einsum optimization only supports einsum ints up to 63")
        }
        let (mut result, new_input_ints) =
            self.optimize_pre_contract_subsets(dont_contract_subsets_into);
        // this is ugly, todo fix
        if new_input_ints.len() == 1 && result.len() > 1 {
            result.pop();
        } else if new_input_ints.len() > 2 {
            result.pop();
            let new_spec = EinsumSpec {
                input_ints: new_input_ints
                    .iter()
                    .copied()
                    .map(bitmask_to_ints)
                    .collect(),
                output_ints: self.output_ints.clone(),
                int_sizes: self.int_sizes.clone(),
            };
            result.extend(new_spec.optimize_dp(check_outer, mem_cap, hash_map_cap)?);
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

// non python methods
impl EinsumSpec {
    pub fn int_map(&self, ints_map: &HashMap<usize, usize>) -> EinsumSpec {
        let mut new_int_sizes: Vec<usize> = vec![0; *ints_map.values().max().unwrap_or(&0) + 1];
        for (k, v) in ints_map {
            new_int_sizes[*v] = self.int_sizes[*k];
        }
        return EinsumSpec {
            input_ints: self
                .input_ints
                .iter()
                .map(|operand_ints| {
                    operand_ints
                        .iter()
                        .map(|inty| *ints_map.get(inty).unwrap())
                        .collect()
                })
                .collect(),
            output_ints: self
                .output_ints
                .iter()
                .map(|inty| *ints_map.get(inty).unwrap())
                .collect(),
            int_sizes: new_int_sizes,
        };
    }
}

static OPT_EINSUM_CACHE: Lazy<
    Mutex<
        HashMap<
            (
                EinsumSpec,
                Option<bool>,
                Option<usize>,
                Option<usize>,
                Option<Vec<usize>>,
            ),
            Vec<Vec<usize>>,
        >,
    >,
> = Lazy::new(|| Mutex::new(HashMap::default()));
#[pyfunction]
pub fn optimize_einsum_spec_cached(
    spec: EinsumSpec,
    check_outer: Option<bool>,
    mem_cap: Option<usize>,
    hash_map_cap: Option<usize>,
    dont_contract_subsets_into: Option<Vec<usize>>,
) -> Result<Vec<Vec<usize>>> {
    let key = (
        spec.clone(),
        check_outer.clone(),
        mem_cap.clone(),
        hash_map_cap.clone(),
        dont_contract_subsets_into.clone(),
    );
    if let Some(r) = OPT_EINSUM_CACHE.lock().unwrap().get(&key) {
        return Ok(r.clone());
    }
    let result = spec.optimize(
        check_outer,
        mem_cap,
        hash_map_cap,
        dont_contract_subsets_into,
    );
    if let Ok(r) = &result {
        OPT_EINSUM_CACHE.lock().unwrap().insert(key, r.clone());
    }
    result
}
