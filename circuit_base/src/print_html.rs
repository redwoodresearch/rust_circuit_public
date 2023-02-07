use std::{
    fmt::{Debug, Write},
    hash::{Hash, Hasher},
};

use anyhow::{anyhow, bail, Context, Result};
use itertools::Itertools;
use pyo3::prelude::*;
use rand::Rng;
use rr_util::{fn_struct, name::Name, py_types::MaybeNotSet, simple_default};
use rustc_hash::FxHashSet as HashSet;

use crate::{
    opaque_iterative_matcher::{get_opaque_type_matcher, OpaqueIterativeMatcherVal},
    prelude::*,
    print::{
        circuit_to_computability_color_code, circuit_to_size_color_code, get_child_comments,
        op_traversal_per, write_comment, CircuitCommenter, PrintOptions,
    },
    CircuitType, HashBytes,
};

#[derive(FromPyObject, Clone)]
pub enum PrintOptionsBase {
    PrintOptions(PrintOptions),
    PrintHtmlOptions(PrintHtmlOptions),
}

impl PrintOptionsBase {
    pub fn print(&self, circuits: Vec<CircuitRc>) -> Result<()> {
        match self {
            PrintOptionsBase::PrintOptions(p) => p.print_circuits(circuits),
            PrintOptionsBase::PrintHtmlOptions(p) => p.print(circuits),
        }
    }
}

simple_default!(PrintOptionsBase {
    PrintOptionsBase::PrintOptions(Default::default())
});

fn_struct!(pub CircuitToColor:Fn(circuit:CircuitRc)->Option<String>);

#[pyclass]
#[derive(Clone)]
pub struct PrintHtmlOptions {
    #[pyo3(get, set)]
    pub shape_only_when_necessary: bool,
    pub traversal: Option<OpaqueIterativeMatcherVal>,
    #[pyo3(get, set)]
    pub display_copy_button: bool,
    #[pyo3(get, set)]
    pub colorer: CircuitToColor,
    #[pyo3(get, set)]
    pub primary_color: String,
    #[pyo3(get, set)]
    pub number_leaves: bool,
    #[pyo3(get, set)]
    pub keep_all_cells: bool,
    pub overall_traversal: Option<OpaqueIterativeMatcherVal>,
    #[pyo3(get, set)]
    pub comment_arg_names: bool,
    #[pyo3(get, set)]
    pub commenters: Vec<CircuitCommenter>,
}

impl Default for PrintHtmlOptions {
    fn default() -> Self {
        Self {
            shape_only_when_necessary: true,
            traversal: Some(OpaqueIterativeMatcherVal::for_end_depth(3)),
            display_copy_button: true,
            colorer: PrintHtmlOptions::type_colorer(),
            primary_color: "lightgray".to_owned(),
            number_leaves: false,
            keep_all_cells: true,
            overall_traversal: Some(OpaqueIterativeMatcherVal::never()),
            comment_arg_names: true,
            commenters: vec![],
        }
    }
}

impl PrintHtmlOptions {
    pub fn fill_overall_data(
        &self,
        circuits: Vec<CircuitRc>,
        nodes_string: &mut String,
        seen: &mut HashSet<HashBytes>,
    ) -> Result<()> {
        fn recurse(
            circ: CircuitRc,
            nodes_string: &mut String,
            seen: &mut HashSet<HashBytes>,
            selfy: &PrintHtmlOptions,
            overall_traversal: Option<OpaqueIterativeMatcherVal>,
            node_comments: Vec<String>,
        ) -> Result<()> {
            if !seen.insert(circ.info().hash) {
                return Ok(());
            }
            let mut desc: String = selfy.repr_line(circ.clone())?;
            write_comment(&mut desc, node_comments, &selfy.commenters, circ.crc())?;
            make_str_safe(&mut desc)?;
            let mut color = match selfy
                .colorer
                .call(circ.clone())
                .context("colorer errored in html printer")?
            {
                Some(col) => col,
                None => selfy.primary_color.clone(),
            };
            make_str_css_safe(&mut color)?;
            let mut name = Name::string_maybe_empty(circ.info().name);
            make_str_safe(&mut name)?;
            write!(
                nodes_string,
                "\"{}\":{{name:\"{}\",desc:\"{}\",color:\"{}\",children:[",
                &circ.hash_base16(),
                name,
                desc,
                &color,
            )
            .unwrap();

            for (child_idx, c) in circ.children().enumerate() {
                write!(nodes_string, "[\"{}\"", c.hash_base16()).unwrap();
                circ.as_module().map(|m| {
                    child_idx.checked_sub(1).map(|i| {
                        let arg_spec = &m.spec.arg_specs[i];
                        arg_spec
                            .symbol
                            .info()
                            .name
                            .map(|n| write!(nodes_string, ",\" ! '{}'\"", n));
                    });
                });
                write!(nodes_string, "],").unwrap();
            }
            nodes_string.push_str("]},");

            let new_traversal_per = if let Some(new_traversal_per) =
                op_traversal_per(overall_traversal, circ.crc())
                    .context("traversal errored in html printer")?
            {
                new_traversal_per
            } else {
                // all finished case
                return Ok(());
            };
            for (i, (child, new_traversal)) in circ.children().zip(new_traversal_per).enumerate() {
                let child_specific_commenters = if selfy.comment_arg_names {
                    get_child_comments(&(**circ), i)
                } else {
                    vec![]
                };
                recurse(
                    child,
                    nodes_string,
                    seen,
                    selfy,
                    new_traversal,
                    child_specific_commenters,
                )?;
            }
            Ok(())
        }
        for circuit in circuits.clone() {
            recurse(
                circuit,
                nodes_string,
                seen,
                &self,
                self.overall_traversal.clone(),
                vec![],
            )?;
        }
        Ok(())
    }
    pub fn fill_to_expand_data(
        &self,
        circuits: Vec<CircuitRc>,
        to_expand_string: &mut String, // Nodes you can see the children of
        seen: &HashSet<HashBytes>,     /* Data of all nodes which you might want to see */
    ) -> Result<()> {
        let mut expanded_nodes: HashSet<HashBytes> = Default::default(); // Nodes that are seen from the beginning
        fn recurse(
            circ: CircuitRc,
            to_expand_string: &mut String, // Nodes you can see the children of
            expanded_nodes: &mut HashSet<HashBytes>,
            selfy: &PrintHtmlOptions,
            traversal: Option<OpaqueIterativeMatcherVal>,
            seen: &HashSet<HashBytes>,
        ) -> Result<()> {
            let key = circ.info().hash;
            if !expanded_nodes.insert(key) {
                return Ok(());
            }
            let new_traversal_per_op = op_traversal_per(traversal, circ.crc())?;

            let new_traversal_per = if let Some(new_traversal_per) = new_traversal_per_op {
                new_traversal_per
            } else {
                // all finished case
                return Ok(());
            };
            write!(to_expand_string, "\"{}\",", &circ.hash_base16()).unwrap();

            for (child, new_traversal) in circ.children().zip(new_traversal_per) {
                recurse(
                    child,
                    to_expand_string,
                    expanded_nodes,
                    selfy,
                    new_traversal.clone(),
                    seen,
                )?;
            }
            Ok(())
        }
        for circuit in circuits {
            recurse(
                circuit,
                to_expand_string,
                &mut expanded_nodes,
                &self,
                self.traversal.clone(),
                seen,
            )?;
        }
        Ok(())
    }

    fn get_print_options(&self) -> PrintOptions {
        PrintOptions {
            bijection: false,
            shape_only_when_necessary: self.shape_only_when_necessary,
            ..Default::default() // Other params are not used in repr_line_info
        }
    }
}

#[pymethods]
impl PrintHtmlOptions {
    #[new]
    #[pyo3(signature=(
        shape_only_when_necessary = PrintHtmlOptions::default().shape_only_when_necessary,
        traversal = PrintHtmlOptions::default().traversal.unwrap(),
        display_copy_button = PrintHtmlOptions::default().display_copy_button,
        colorer = PrintHtmlOptions::default().colorer,
        primary_color = PrintHtmlOptions::default().primary_color,
        number_leaves = PrintHtmlOptions::default().number_leaves,
        keep_all_cells = PrintHtmlOptions::default().keep_all_cells,
        overall_traversal = PrintHtmlOptions::default().overall_traversal.unwrap(),
        comment_arg_names = PrintOptions::default().comment_arg_names,
        commenters = PrintOptions::default().commenters
    ))] // None are mapped to the default
    pub fn py_new(
        shape_only_when_necessary: bool,
        traversal: Option<OpaqueIterativeMatcherVal>,
        display_copy_button: bool,
        colorer: CircuitToColor,
        primary_color: String,
        number_leaves: bool,
        keep_all_cells: bool,
        overall_traversal: Option<OpaqueIterativeMatcherVal>,
        comment_arg_names: bool,
        commenters: Vec<CircuitCommenter>,
    ) -> Result<Self> {
        let result = Self {
            shape_only_when_necessary,
            traversal,
            display_copy_button,
            colorer,
            primary_color,
            number_leaves,
            keep_all_cells,
            overall_traversal,
            comment_arg_names,
            commenters,
        };
        Ok(result)
    }

    #[pyo3(signature=(
        shape_only_when_necessary = Default::default(),
        traversal = Default::default(),
        display_copy_button = Default::default(),
        colorer = MaybeNotSet(None),
        primary_color = Default::default(),
        number_leaves = Default::default(),
        keep_all_cells = Default::default(),
        overall_traversal = Default::default(),
        comment_arg_names = Default::default(),
        commenters = Default::default()
    ))]
    pub fn evolve(
        &self,
        shape_only_when_necessary: MaybeNotSet<bool>,
        traversal: MaybeNotSet<Option<OpaqueIterativeMatcherVal>>,
        display_copy_button: MaybeNotSet<bool>,
        colorer: MaybeNotSet<CircuitToColor>,
        primary_color: MaybeNotSet<String>,
        number_leaves: MaybeNotSet<bool>,
        keep_all_cells: MaybeNotSet<bool>,
        overall_traversal: MaybeNotSet<Option<OpaqueIterativeMatcherVal>>,
        comment_arg_names: MaybeNotSet<bool>,
        commenters: MaybeNotSet<Vec<CircuitCommenter>>,
    ) -> Self {
        let cloned = self.clone();
        Self {
            shape_only_when_necessary: shape_only_when_necessary
                .0
                .unwrap_or(cloned.shape_only_when_necessary),
            traversal: traversal.0.unwrap_or(cloned.traversal),
            display_copy_button: display_copy_button.0.unwrap_or(cloned.display_copy_button),
            colorer: colorer.0.unwrap_or(cloned.colorer),
            primary_color: primary_color.0.unwrap_or(cloned.primary_color),
            number_leaves: number_leaves.0.unwrap_or(cloned.number_leaves),
            keep_all_cells: keep_all_cells.0.unwrap_or(cloned.keep_all_cells),
            overall_traversal: overall_traversal.0.unwrap_or(cloned.overall_traversal),
            comment_arg_names: comment_arg_names.0.unwrap_or(cloned.comment_arg_names),
            commenters: commenters.0.unwrap_or(cloned.commenters),
        }
    }

    #[getter(traversal)]
    pub fn py_get_traversal(&self) -> PyObject {
        OpaqueIterativeMatcherVal::op_to_object(&self.traversal)
    }

    #[setter(traversal)]
    pub fn py_set_traversal(&mut self, traversal: Option<OpaqueIterativeMatcherVal>) {
        self.traversal = traversal
    }

    #[getter(overall_traversal)]
    pub fn py_get_overall_traversal(&self) -> PyObject {
        OpaqueIterativeMatcherVal::op_to_object(&self.overall_traversal)
    }

    #[setter(overall_traversal)]
    pub fn py_set_overall_traversal(
        &mut self,
        overall_traversal: Option<OpaqueIterativeMatcherVal>,
    ) {
        self.overall_traversal = overall_traversal
    }

    #[pyo3(signature=(*circuits))]
    pub fn print(&self, circuits: Vec<CircuitRc>) -> Result<()> {
        let html = self.repr(circuits)?;
        Python::with_gil(|py| -> Result<()> {
            let fun: Py<PyAny> = PyModule::from_code(
                py,
                &format!(
                    "def run():
                    from IPython.display import HTML, display
                    display(HTML(\"\"\"{html}\"\"\"))"
                ), // Dangerous, but " were escaped so it should be fine?
                "",
                "",
            )?
            .getattr("run")?
            .into();
            fun.call0(py)?;
            Ok(())
        })
    }

    #[pyo3(signature=(*circuits))]
    pub fn repr(&self, circuits: Vec<CircuitRc>) -> Result<String> {
        let mut html = HTML_HEADER.to_owned();
        let mut primary_color = self.primary_color.clone();
        make_str_css_safe(&mut primary_color)?;

        // var data = {1: {
        //     name: 'I k2 logits_diff, true I',
        //     desc: 'I k2 logits_diff, true I [t: Cumulant] [s: ()]',
        //     color: 'red',
        //     children: [['2', ' ! input 1'], ['3']],
        //   }

        html.push_str("var datatoggle___js___ = {");
        let mut seen: HashSet<HashBytes> = HashSet::default();
        self.fill_overall_data(circuits.clone(), &mut html, &mut seen)?;
        html.push_str("};");

        // var toExpandtoggle___js___ = ['1', ...];
        html.push_str("var toExpandtoggle___js___ = [");
        self.fill_to_expand_data(circuits.clone(), &mut html, &seen)?;
        html.push_str("];");

        // var rootstoggle___js___ = ['1', ...];
        html.push_str("var rootstoggle___js___ = [");
        html.push_str(
            &circuits
                .iter()
                .map(|c| format!("\"{}\"", &c.hash_base16()))
                .join(","),
        );
        html.push_str("];");

        html.push_str(if self.display_copy_button {
            "var displayCopyButtontoggle___js___ = true;"
        } else {
            "var displayCopyButtontoggle___js___ = false;"
        });
        html.push_str(if self.number_leaves {
            "var numberLeaf___js___ = true;"
        } else {
            "var numberLeaf___js___ = false;"
        });

        // Jupyter just keeps adding new html elements, so we have to create divs
        // with unique ids. Seed is the suffix of every id created in one block.
        let mut rng = rand::thread_rng();
        let seed = rng.gen_range(0..100000000);
        html.push_str(&format!("var seedtoggle___js___ = \"{seed}\";"));

        html.push_str(CSS_HEADER);

        html.push_str(&format!("--primary: {};", primary_color,));
        html.push_str(CSS);

        html.push_str(&format!(
            "<div class=\"mainDiv\" id=\"s{seed}\"><ul class=\"list\"></ul></div>",
        ));
        html.push_str(JS);

        if self.keep_all_cells {
            // If keep_all_cells, variables in all cells need to have different names.
            // To do so, the seed needs to be in variable names which shouldn't be overwritten.
            // These variables have ___js___ as suffix.
            // Note: conflict with circuit name could be problematic.
            html = html.replace("___js___", &format!("{seed}"));
        }

        Ok(html)
    }

    pub fn repr_line(&self, circ: CircuitRc) -> Result<String> {
        let line_info = self.get_print_options().repr_line_info(circ.clone());
        if let Some(name) = circ.info().name {
            Ok(format!("'{name}'{}", &line_info?))
        } else {
            line_info
        }
    }

    pub fn light(&self) -> Self {
        self.evolve(
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(Some("#303030".to_owned())), // darkgray isn't actually dark .-.
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(None),
            MaybeNotSet(None),
        )
    }

    #[staticmethod]
    #[pyo3(signature=(*circuits, depth = 3))]
    pub fn print_depth(circuits: Vec<CircuitRc>, depth: usize) -> Result<()> {
        PrintHtmlOptions {
            traversal: Some(OpaqueIterativeMatcherVal::for_end_depth(depth)),
            overall_traversal: Some(OpaqueIterativeMatcherVal::never()),
            ..Default::default()
        }
        .print(circuits)
    }

    #[staticmethod]
    pub fn type_nest_default(
        circuit_type: CircuitType,
        traversal: Option<OpaqueIterativeMatcherVal>,
    ) -> Self {
        Self {
            number_leaves: true,
            traversal: Some(get_opaque_type_matcher(circuit_type, traversal)),
            ..Default::default()
        }
    }

    #[staticmethod]
    pub fn einsum_nest_default(traversal: Option<OpaqueIterativeMatcherVal>) -> Self {
        Self::type_nest_default(CircuitType::Einsum, traversal)
    }

    #[staticmethod]
    pub fn add_nest_default(traversal: Option<OpaqueIterativeMatcherVal>) -> Self {
        Self::type_nest_default(CircuitType::Add, traversal)
    }

    #[staticmethod]
    pub fn size_colorer() -> CircuitToColor {
        CircuitToColor::new_dyn(Box::new(|c| {
            Ok(circuit_to_size_color_code(c).map(|c| color_from_colorcode(c)))
        }))
    }

    #[staticmethod]
    pub fn type_colorer() -> CircuitToColor {
        CircuitToColor::new_dyn(Box::new(|circuit: CircuitRc| {
            Ok(Some(match &(**circuit) {
                Circuit::Einsum(_) => color_from_colorcode(0),
                Circuit::Add(_) => color_from_colorcode(1),
                Circuit::Concat(_) => color_from_colorcode(2),
                Circuit::GeneralFunction(_) => color_from_colorcode(3),
                Circuit::Array(_) => color_from_colorcode(4),
                Circuit::DiscreteVar(_) => color_from_colorcode(5),
                _ => {
                    let mut hasher = rustc_hash::FxHasher::default();
                    circuit.variant_string().hash(&mut hasher);
                    color_from_hash(hasher.finish() as usize)
                }
            }))
        }))
    }

    #[staticmethod]
    pub fn hash_colorer() -> CircuitToColor {
        CircuitToColor::new_dyn(Box::new(|circuit: CircuitRc| {
            Ok(Some(color_from_hash(circuit.info().hash_usize())))
        }))
    }
    #[staticmethod]
    pub fn fixed_color(color: String) -> CircuitToColor {
        CircuitToColor::new_dyn(Box::new(move |_circuit: CircuitRc| Ok(Some(color.clone()))))
    }

    #[staticmethod]
    pub fn computability_colorer() -> CircuitToColor {
        CircuitToColor::new_dyn(Box::new(|c| {
            Ok(circuit_to_computability_color_code(c).map(|c| color_from_colorcode(c)))
        }))
    }
}

fn color_from_hash(hash: usize) -> String {
    let hue = (hash % (360 / 20)) * 20;
    return format!("hsl({hue}, 90%, 60%)");
}

const COLORCODE_TO_RGB: [&str; 6] = [
    "rgb(222,56,43)",
    "rgb(57,181,74)",
    "rgb(255,199,6)",
    "rgb(0,111,184)",
    "rgb(118,38,113)",
    "rgb(44,181,233)",
];

fn color_from_colorcode(code: usize) -> String {
    assert!(code < COLORCODE_TO_RGB.len(), "invalid colorcode");
    return COLORCODE_TO_RGB[code].to_owned();
}

// this does mutation to avoid copying in the 99% case when it doesn't change anything
fn make_str_safe(s: &mut String) -> Result<()> {
    if s.contains("___js___") {
        bail!(anyhow!("___js___ is a reserved name. Don't use it."))
    }
    if s.contains("\"") {
        bail!(anyhow!("Can't use quotes in string printed to html."))
    }
    if s.contains("<") || s.contains(">") {
        let newstring = s.replace("<", "&lt").replace(">", "&gt");
        s.truncate(0);
        s.push_str(&newstring);
    }
    Ok(())
}
fn make_str_css_safe(s: &mut String) -> Result<()> {
    make_str_safe(s).context("wrong color format (no quotes allowed)")?;
    if s.contains(";") {
        let newstring = s.replace(";", "");
        s.truncate(0);
        s.push_str(&newstring);
    }
    return Ok(());
}

// Variables that shouldn't be overwritten between cells should have ___js___ as suffix.
const HTML_HEADER: &str = r#"
  <script>
    var expandClonestoggle___js___ = true;"#;
const CSS_HEADER: &str = r#"
  </script>
  <style>
    :root {"#;
const CSS: &str = r#"
    }
    .mainDiv {
      margin: 0;
      padding: 0;
    }
    .list.list {
      margin: 0;
      padding: 0;
      list-style-type: none;
      color: var(--primary);
    }
    .expand-caret.expand-caret.expand-caret,
    .expand-caret-end.expand-caret-end.expand-caret-end {
      cursor: pointer;
      -webkit-user-select: none; /* Safari 3.1+ */
      -moz-user-select: none; /* Firefox 2+ */
      -ms-user-select: none; /* IE 10+ */
      user-select: none;
      margin-left: 0.2em;
      margin-right: 0.4em;
    }
    .expand-caret::before,
    .expand-caret-end::before {
      content: '>';
      color: var(--primary);
      display: inline-block;
    }
    .expand-caret-end::before {
      color: var(--primary);
      opacity: 0.5;
    }
    .expand-caret-down::before {
      -ms-transform: rotate(90deg);
      -webkit-transform: rotate(90deg);
      transform: rotate(90deg);
      color: var(--primary);
    }
    .nested.nested.nested {
      display: none;
    }
    .active.active.active {
      padding-left: 0.2em;
      margin-left: 0.42em;

      display: block;
    }
    .active.active.onlychild {
      margin-left: 0em;
      padding-left: 0em;
    }
    .copy-button {
      margin: 0px 0.2em 0px 0.2em;
      padding: 0px;
      background-color: transparent;
      border-radius: 3px;
      border-color: transparent;
      outline: none;
      opacity: 0.6;
    }
    .copy-button:hover {
      opacity: 1;
    }
    .copy-button:active {
      background-color: var(--primary);
    }
    .copy-button:focus {
      outline: none;
    }
    .inline-pre {
      margin: 0;
      padding: 0;
      display: inline;
    }
    .bold {
      text-shadow: 1px 1px 0px;
    }
  </style>"#;
const JS: &str = r#"
<script>
  function getElement(id) {
    return document.getElementById(id);
  }

  function getLastChild(div) {
    return div.children[div.children.length - 1];
  }

  // Parameters: id of the parent html element, id of the node to expand
  function addChildtoggle___js___(parent_id, id, child_info) {
    var child = document.createElement('li');
    child.classList.add('list');

    // Example of the generated inner HTML:
    // <span class="expand-caret" onclick="toggle('1')"></span>
    // <pre class="inline-pre" style="color: hsl(70, 100%, 70%)">1 'I k2 logits_diff, true I' [t: Cumulant] [s: ()]</pre>
    // <pre class="inline-pre" style="color: magenta"></pre>
    // <button onclick="toClipboard('I k2 logits_diff, true I')" class="copy-button">📋</button>
    // <ul class="nested list" style="border-left: 3px solid hsl(70, 100%, 70%)"></ul>;

    const childData = datatoggle___js___[id];
    if (!childData.displayedCount) {
      childData.displayedCount = 0;
    }
    var elementId = id + '@' + childData.displayedCount + '@' + seedtoggle___js___;
    child.id = elementId; // Assumes no "@" in ID

    var desc = childData.desc + child_info;
    if (childData.displayedCount > 0) {
      desc += ' (repeat)';
    }
    var copyButton = displayCopyButtontoggle___js___
      ? `<button onclick="toClipboard('` + childData.name + `')" class="copy-button">📋</button>`
      : '';

    var onlyChild = datatoggle___js___[id].children.length == 0;
    var expandForbidden = !expandClonestoggle___js___ && childData.displayedCount > 0;
    var cantExpand = onlyChild || expandForbidden;
    var caretClass = cantExpand ? 'expand-caret-end' : 'expand-caret';

    var innerHTML =
      `<span class="` +
      caretClass +
      `" onclick="toggletoggle___js___('` +
      elementId +
      `')"></span><pre class="inline-pre" style="color: ` +
      childData.color +
      `">` +
      desc +
      `</pre><pre class="inline-pre" style="color:magenta"></pre>` +
      copyButton +
      `<ul class="nested list" style="border-left: 3px solid ` +
      childData.color +
      `"></ul>`;
    child.innerHTML = innerHTML;
    child.children[1].addEventListener('mouseenter', function () {
      toggleSiblingsBoldtoggle___js___(elementId);
    });
    child.children[1].addEventListener('mouseleave', function () {
      toggleSiblingsBoldtoggle___js___(elementId);
    });
    childData.displayedCount += 1;
    getLastChild(getElement(parent_id)).appendChild(child);
  }

  function toClipboard(copyText) {
    navigator.clipboard.writeText(copyText);
  }
  function toggletoggle___js___(elementId) {
    var elt = getElement(elementId);
    console.log(elt, elementId);
    var nodeId = elementId.split('@')[0];

    var noChild = datatoggle___js___[nodeId].children.length == 0;
    var expandAllowed = expandClonestoggle___js___ || elementId.split('@')[1] == '0';
    if (!noChild && expandAllowed) {
      getLastChild(elt).classList.toggle('active');
      elt.children[0].classList.toggle('expand-caret-down');

      let alreadExpanded = getLastChild(elt).children.length > 0;
      if (!alreadExpanded) {
        datatoggle___js___[nodeId].children.forEach(function (childIdAndInfo) {
          var childId = childIdAndInfo[0];
          var childInfo = childIdAndInfo.slice(1).join(' ');
          addChildtoggle___js___(elementId, childId, childInfo);
        });
      }
    }
  }
  function toggleSiblingsBoldtoggle___js___(elementId) {
    var nodeId = elementId.split('@')[0];
    if (!datatoggle___js___[nodeId].displayedCount) return;
    var thisElementCount = elementId.split('@')[1];
    for (var i = 0; i < datatoggle___js___[nodeId].displayedCount; i++) {
      if (i != thisElementCount) {
        var otherElementId = nodeId + '@' + i + '@' + seedtoggle___js___;
        getElement(otherElementId).children[1].classList.toggle('bold');
      }
    }
  }
  function updateLeafNumbers___js___() {
    if (!numberLeaf___js___) return;
    document.querySelectorAll('#s' + seedtoggle___js___ + ' .nested.active').forEach(function (d, i) {
      d.parentNode.children[2].innerText = '';
    });
    document.querySelectorAll('#s' + seedtoggle___js___ + ' .nested:empty').forEach(function (d, i) {
      d.parentNode.children[2].innerText = ' # ' + i;
    });
  }
  
  rootstoggle___js___.forEach(function (root) {
    console.log(root);
    addChildtoggle___js___("s"+seedtoggle___js___, root, '');
  });
  toExpandtoggle___js___.forEach(function (expandElt) {
    console.log(expandElt);
    toggletoggle___js___(expandElt + '@0@' + seedtoggle___js___);
  });
  updateLeafNumbers___js___();
</script>
"#;
