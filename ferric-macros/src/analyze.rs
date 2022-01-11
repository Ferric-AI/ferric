// Copyright 2022 The Ferric AI Project Developers
use std::collections::HashMap;
use syn::{Error, Expr, Ident};

use crate::parse::ModelAst;

/// VariableIR is the Intermediate Representation of a random variable.
pub struct VariableIR {
    pub var_ident: Ident,
    pub type_ident: Ident,
    pub dependency: Expr,
    pub is_queried: bool,
    pub is_observed: bool,
}

/// ModelIR is the Intermediate Representation of the model before code generation.
/// This represents the output of the analyze phase in a proc_macro pipeline.
pub struct ModelIR {
    pub model_ident: Ident,
    pub use_exprs: Vec<Expr>,
    pub variables: HashMap<String, VariableIR>,
}

pub fn analyze(ast: ModelAst) -> Result<ModelIR, Error> {
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
            dependency: stmt.dependency,
            type_ident: stmt.type_ident,
            is_queried: false,
            is_observed: false,
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
                ))
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
                ))
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
    Ok(ModelIR {
        model_ident: ast.model_ident,
        use_exprs: ast.use_exprs,
        variables,
    })
}

#[test]
fn test_analyze_errors() {
    use quote::quote;
    use syn::parse2;

    // duplicate definition of variable
    assert!(analyze(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;
            let rain : bool ~ Bernoulli::new( 0.2 );
            let rain : bool ~ Bernoulli::new( 0.2 );
        ))
        .unwrap()
    )
    .is_err());

    // undefined query variable
    assert!(analyze(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;
            let rain : bool ~ Bernoulli::new( 0.2 );
            query sprinkler;
        ))
        .unwrap()
    )
    .is_err());

    // duplicate query
    assert!(analyze(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;
            let rain : bool ~ Bernoulli::new( 0.2 );
            query rain;
            query rain;
        ))
        .unwrap()
    )
    .is_err());

    // undefined observe
    assert!(analyze(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;
            let rain : bool ~ Bernoulli::new( 0.2 );
            observe sprinkler;
        ))
        .unwrap()
    )
    .is_err());

    // duplicate observe
    assert!(analyze(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;
            let rain : bool ~ Bernoulli::new( 0.2 );
            observe rain;
            observe rain;
        ))
        .unwrap()
    )
    .is_err());
}

#[test]
fn test_analyze_output() {
    use quote::quote;
    use syn::{parse2, parse_quote};

    let model_ast = parse2::<ModelAst>(quote!(
        mod grass;
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
    let exp_type_name: Ident = parse_quote!(bool);
    let exp_dependency: &Expr = &parse_quote!(Bernoulli::new(0.2));
    assert_eq!(var.var_ident, exp_var_name);
    assert_eq!(var.type_ident, exp_type_name);
    assert_eq!(var.dependency, *exp_dependency);
    assert!(var.is_queried);
    assert!(!var.is_observed);
}
