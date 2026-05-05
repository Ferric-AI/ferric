// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::TokenStream;
use std::collections::{HashMap, HashSet};
use syn::{Error, Expr, Ident, Type};

use crate::parse::ModelAst;

/// ConstantIR is the Intermediate Representation of a model constant.
pub struct ConstantIR {
    pub const_ident: Ident,
    pub type_ident: Type,
    pub order: usize,
}

/// Indexed random-variable dimension.
#[derive(Clone)]
pub struct IndexRangeIR {
    pub index_ident: Ident,
    pub upper_ident: Ident,
}

/// VariableIR is the Intermediate Representation of a random variable.
pub struct VariableIR {
    pub var_ident: Ident,
    pub indices: Vec<IndexRangeIR>,
    pub type_ident: Type,
    pub dependency: TokenStream,
    pub max_expr: Option<Expr>,
    pub max_const_ident: Option<Ident>,
    /// `true` if defined with `~` (distribution), `false` if defined with `=` (deterministic expr).
    pub is_stochastic: bool,
    pub is_queried: bool,
    pub is_observed: bool,
    pub order: usize,
}

/// ModelIR is the Intermediate Representation of the model before code generation.
/// This represents the output of the analyze phase in a proc_macro pipeline.
pub struct ModelIR {
    pub model_ident: Ident,
    pub use_exprs: Vec<Expr>,
    pub consts: HashMap<String, ConstantIR>,
    pub variables: HashMap<String, VariableIR>,
}

pub fn analyze(ast: ModelAst) -> Result<ModelIR, Error> {
    let mut consts = HashMap::<String, ConstantIR>::new();
    for konst in ast.consts.into_iter() {
        let const_name = konst.const_ident.to_string();
        if consts.contains_key(&const_name) {
            return Err(Error::new(
                konst.const_ident.span(),
                format!("duplicate declaration of constant `{}`", const_name),
            ));
        }
        consts.insert(
            const_name,
            ConstantIR {
                const_ident: konst.const_ident,
                type_ident: konst.type_ident,
                order: konst.order,
            },
        );
    }

    let mut variables = HashMap::<String, VariableIR>::new();
    // analyze all the statements
    for stmt in ast.stmts.into_iter() {
        let var_name = stmt.var_ident.to_string();
        // the variable shouldn't have been previously defined
        if variables.contains_key(&var_name) {
            return Err(Error::new(
                stmt.var_ident.span(),
                format!("duplicate declaration of variable `{}`", var_name),
            ));
        }
        let variable = VariableIR {
            var_ident: stmt.var_ident,
            indices: stmt
                .indices
                .into_iter()
                .map(|idx| IndexRangeIR {
                    index_ident: idx.index_ident,
                    upper_ident: idx.upper_ident,
                })
                .collect(),
            dependency: stmt.dependency,
            max_expr: stmt.max_expr,
            max_const_ident: stmt.max_const_ident,
            type_ident: stmt.type_ident,
            is_stochastic: stmt.is_stochastic,
            is_queried: false,
            is_observed: false,
            order: stmt.order,
        };
        variables.insert(var_name, variable);
    }
    // analyze the query statements
    for query in ast.queries.into_iter() {
        let var_name = query.to_string();
        match variables.get_mut(&var_name) {
            None => {
                return Err(Error::new(
                    query.span(),
                    format!("undefined query variable `{}`", var_name),
                ));
            }
            Some(variable) => {
                if variable.is_queried {
                    return Err(Error::new(
                        query.span(),
                        format!("duplicate query of variable `{}`", var_name),
                    ));
                } else {
                    variable.is_queried = true;
                }
            }
        }
    }
    // analyze the observe statements
    for obs in ast.observes.into_iter() {
        let var_name = obs.to_string();
        match variables.get_mut(&var_name) {
            None => {
                return Err(Error::new(
                    obs.span(),
                    format!("undefined observed variable `{}`", var_name),
                ));
            }
            Some(variable) => {
                if variable.is_observed {
                    return Err(Error::new(
                        obs.span(),
                        format!("duplicate observe of variable `{}`", var_name),
                    ));
                } else {
                    variable.is_observed = true;
                }
            }
        }
    }

    validate_indices_and_mark_max_bounds(&mut variables, &consts)?;

    Ok(ModelIR {
        model_ident: ast.model_ident,
        use_exprs: ast.use_exprs,
        consts,
        variables,
    })
}

fn validate_indices_and_mark_max_bounds(
    variables: &mut HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> Result<(), Error> {
    let mut stochastic_bounds = HashSet::<String>::new();

    for variable in variables.values() {
        validate_max_annotation(variable, consts)?;
        validate_index_names(variable)?;

        for index in variable.indices.iter() {
            let upper_name = index.upper_ident.to_string();
            if let Some(konst) = consts.get(&upper_name) {
                if konst.order >= variable.order {
                    return Err(Error::new(
                        index.upper_ident.span(),
                        format!(
                            "index upper bound `{}` must be declared before `{}`",
                            upper_name, variable.var_ident
                        ),
                    ));
                }
                continue;
            }

            match variables.get(&upper_name) {
                None => {
                    return Err(Error::new(
                        index.upper_ident.span(),
                        format!("undefined index upper bound `{}`", upper_name),
                    ));
                }
                Some(bound_var) => {
                    if bound_var.order >= variable.order {
                        return Err(Error::new(
                            index.upper_ident.span(),
                            format!(
                                "index upper bound `{}` must be declared before `{}`",
                                upper_name, variable.var_ident
                            ),
                        ));
                    }
                    if bound_var.is_stochastic {
                        stochastic_bounds.insert(upper_name.clone());
                        if bound_var.max_expr.is_none() {
                            return Err(Error::new(
                                bound_var.var_ident.span(),
                                format!(
                                    "stochastic index upper bound `{}` requires an explicit `max ...` annotation",
                                    upper_name
                                ),
                            ));
                        }
                    }
                    if variable.is_observed && !bound_var.is_observed {
                        return Err(Error::new(
                            index.upper_ident.span(),
                            format!(
                                "observed indexed variable `{}` requires index bound `{}` to be observed or constant",
                                variable.var_ident, upper_name
                            ),
                        ));
                    }
                }
            }
        }
    }

    if stochastic_bounds.len() > 2 {
        let span = variables
            .values()
            .flat_map(|v| v.indices.iter())
            .find(|idx| {
                variables
                    .get(&idx.upper_ident.to_string())
                    .map(|v| v.is_stochastic)
                    .unwrap_or(false)
            })
            .map(|idx| idx.upper_ident.span())
            .unwrap_or_else(proc_macro2::Span::call_site);
        return Err(Error::new(
            span,
            "at most two stochastic index-bound variables are currently supported",
        ));
    }

    Ok(())
}

fn validate_index_names(variable: &VariableIR) -> Result<(), Error> {
    let mut names = HashMap::<String, Ident>::new();
    for index in variable.indices.iter() {
        let index_ident = &index.index_ident;
        let name = index_ident.to_string();
        if names.contains_key(&name) {
            return Err(Error::new(
                index_ident.span(),
                format!("duplicate index name `{}`", name),
            ));
        }
        names.insert(name, index_ident.clone());
    }
    Ok(())
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn validate_max_annotation(
    variable: &VariableIR,
    consts: &HashMap<String, ConstantIR>,
) -> Result<(), Error> {
    if variable.max_expr.is_none() {
        return Ok(());
    }
    if !variable.is_stochastic {
        return Err(Error::new(
            variable.var_ident.span(),
            "`max ...` can only be used on stochastic variables",
        ));
    }
    if variable.max_const_ident.is_none() {
        return Ok(());
    }
    let max_ident = variable
        .max_const_ident
        .as_ref()
        .expect("checked max identifier exists");

    if !consts.contains_key(&max_ident.to_string()) {
        return Err(Error::new(
            max_ident.span(),
            "`max ...` must be a literal or a previously declared constant",
        ));
    }

    let konst = consts
        .get(&max_ident.to_string())
        .expect("checked max constant exists");
    if konst.order >= variable.order {
        return Err(Error::new(
            max_ident.span(),
            format!(
                "max constant `{}` must be declared before `{}`",
                max_ident, variable.var_ident
            ),
        ));
    }
    Ok(())
}

#[test]
fn test_analyze_errors() {
    use quote::quote;
    use syn::parse2;

    // duplicate definition of variable
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name grass;
                use ferric::distributions::Bernoulli;
                let rain : bool ~ Bernoulli::new( 0.2 );
                let rain : bool ~ Bernoulli::new( 0.2 );
            ))
            .unwrap()
        )
        .is_err()
    );

    // undefined query variable
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name grass;
                use ferric::distributions::Bernoulli;
                let rain : bool ~ Bernoulli::new( 0.2 );
                query sprinkler;
            ))
            .unwrap()
        )
        .is_err()
    );

    // duplicate query
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name grass;
                use ferric::distributions::Bernoulli;
                let rain : bool ~ Bernoulli::new( 0.2 );
                query rain;
                query rain;
            ))
            .unwrap()
        )
        .is_err()
    );

    // undefined observe
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name grass;
                use ferric::distributions::Bernoulli;
                let rain : bool ~ Bernoulli::new( 0.2 );
                observe sprinkler;
            ))
            .unwrap()
        )
        .is_err()
    );

    // duplicate observe
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name grass;
                use ferric::distributions::Bernoulli;
                let rain : bool ~ Bernoulli::new( 0.2 );
                observe rain;
                observe rain;
            ))
            .unwrap()
        )
        .is_err()
    );

    // duplicate const
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                const n: u64;
                const n: u64;
            ))
            .unwrap()
        )
        .is_err()
    );

    // undefined index bound
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let x[i of n] : bool ~ Bernoulli::new(0.5);
            ))
            .unwrap()
        )
        .is_err()
    );

    // const index bound must be declared first
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let x[i of n] : bool ~ Bernoulli::new(0.5);
                const n : u64;
            ))
            .unwrap()
        )
        .is_err()
    );

    // variable index bound must be declared first
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let x[i of n] : bool ~ Bernoulli::new(0.5);
                let n : u64 ~ Poisson::new(3.0);
            ))
            .unwrap()
        )
        .is_err()
    );

    // observed indexed variables need observed or constant bounds
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let n : u64 ~ Poisson::new(3.0) max 3;
                let x[i of n] : bool ~ Bernoulli::new(0.5);
                observe x;
            ))
            .unwrap()
        )
        .is_err()
    );

    // at most two stochastic index bounds are currently supported
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let a : u64 ~ Poisson::new(3.0) max 3;
                let b : u64 ~ Poisson::new(3.0) max 3;
                let c : u64 ~ Poisson::new(3.0) max 3;
                let x[i of a, j of b, k of c] : bool ~ Bernoulli::new(0.5);
            ))
            .unwrap()
        )
        .is_err()
    );

    // stochastic index bounds require an explicit max annotation
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let n : u64 ~ Poisson::new(3.0);
                let x[i of n] : bool ~ Bernoulli::new(0.5);
            ))
            .unwrap()
        )
        .is_err()
    );

    // named indices cannot be duplicated in the same variable
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                const n : u64;
                const m : u64;
                let x[idx of n, idx of m] : bool ~ Bernoulli::new(0.5);
            ))
            .unwrap()
        )
        .is_err()
    );

    // max annotations must be literals or constants, not random variables
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let limit : u64 ~ Poisson::new(3.0);
                let n : u64 ~ Poisson::new(3.0) max limit;
            ))
            .unwrap()
        )
        .is_err()
    );

    // max constants must be declared before the variable that uses them
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let n : u64 ~ Poisson::new(3.0) max limit;
                const limit : u64;
            ))
            .unwrap()
        )
        .is_err()
    );

    // deterministic variables cannot have max annotations
    assert!(
        analyze(
            parse2::<ModelAst>(quote!(
                name indexed;
                let n : u64 = 3 max 4;
            ))
            .unwrap()
        )
        .is_err()
    );
}

#[test]
fn test_analyze_output() {
    use quote::quote;
    use syn::{parse_quote, parse2};

    let model_ast = parse2::<ModelAst>(quote!(
        name grass;
        use ferric::distributions::Bernoulli;

        let rain : bool ~ Bernoulli::new( 0.2 );

        let sprinkler : bool ~
            if rain {
                Bernoulli::new( 0.01 )
            } else {
                Bernoulli::new( 0.4 )
            };

        let grass_wet : bool ~ Bernoulli::new(
            if sprinkler && rain { 0.99 }
            else if sprinkler && !rain { 0.9 }
            else if !sprinkler && rain { 0.8 }
            else { 0.0 }
        );

        observe grass_wet;
        query rain;
        query sprinkler;
    ))
    .unwrap();

    let model_ir = analyze(model_ast).unwrap();

    let exp_model_name: Ident = parse_quote!(grass);
    assert_eq!(model_ir.model_ident, exp_model_name);

    let exp_use_exprs: &Expr = &parse_quote!(ferric::distributions::Bernoulli);
    assert_eq!(model_ir.use_exprs[0], *exp_use_exprs);
    assert_eq!(model_ir.use_exprs.len(), 1);

    let var = model_ir.variables.get(&String::from("rain")).unwrap();
    let exp_var_name: Ident = parse_quote!(rain);
    let exp_type_name: Type = parse_quote!(bool);
    let exp_dependency: TokenStream = parse_quote!(Bernoulli::new(0.2));
    assert_eq!(var.var_ident, exp_var_name);
    assert_eq!(var.type_ident, exp_type_name);
    assert_eq!(var.dependency.to_string(), exp_dependency.to_string());
    assert!(var.is_stochastic);
    assert!(var.is_queried);
    assert!(!var.is_observed);
}

#[test]
fn test_analyze_indexed_variables() {
    use quote::quote;
    use syn::parse2;

    let model_ir = analyze(
        parse2::<ModelAst>(quote!(
            name indexed;
            const m : u64;
            let n : u64 ~ Poisson::new(3.0) max 3;
            let d : u64 = 2;
            let x[row of n, col of m] : bool ~ Bernoulli::new(0.5);
            let y[item of d] : bool ~ Bernoulli::new(0.5);
            observe n;
            query x;
            query y;
        ))
        .unwrap(),
    )
    .unwrap();

    assert_eq!(model_ir.consts.len(), 1);
    let n_var = model_ir.variables.get("n").unwrap();
    assert!(n_var.is_observed);
    assert!(n_var.max_expr.is_some());
    let x_var = model_ir.variables.get("x").unwrap();
    assert_eq!(x_var.indices.len(), 2);
    assert_eq!(x_var.indices[0].index_ident.to_string(), "row");
    assert!(x_var.is_queried);
    let d_var = model_ir.variables.get("d").unwrap();
    assert!(!d_var.is_stochastic);
    assert!(d_var.max_expr.is_none());
    assert_eq!(model_ir.variables.get("y").unwrap().indices.len(), 1);
}

#[test]
fn test_analyze_deterministic_var() {
    use quote::quote;
    use syn::{parse_quote, parse2};

    let model_ast = parse2::<ModelAst>(quote!(
        name det;
        use ferric::distributions::Bernoulli;

        let x : bool ~ Bernoulli::new(0.5);
        let two_x : u8 = 2u8 * x as u8;

        observe two_x;
        query x;
    ))
    .unwrap();

    let model_ir = analyze(model_ast).unwrap();

    let x_var = model_ir.variables.get(&String::from("x")).unwrap();
    assert!(x_var.is_stochastic);
    assert!(!x_var.is_observed);
    assert!(x_var.is_queried);

    let two_x_var = model_ir.variables.get(&String::from("two_x")).unwrap();
    assert!(!two_x_var.is_stochastic);
    assert!(two_x_var.is_observed);
    assert!(!two_x_var.is_queried);

    let exp_dep: TokenStream = parse_quote!(2u8 * x as u8);
    assert_eq!(two_x_var.dependency.to_string(), exp_dep.to_string());
}
