use anyhow::Result;
use circuit_base::{expand_node::replace_expand_bottom_up, CircuitRc, DiscreteVar};

pub fn discrete_var_sample_all<F>(circuit: CircuitRc, should_sample: F) -> Result<CircuitRc>
where
    F: Fn(&DiscreteVar) -> bool,
{
    replace_expand_bottom_up(circuit, |c| {
        if c.as_discrete_var()
            .map(|x| should_sample(x))
            .unwrap_or(false)
        {
            Some(c.as_discrete_var().unwrap().values().clone())
        } else {
            None
        }
    })
}
