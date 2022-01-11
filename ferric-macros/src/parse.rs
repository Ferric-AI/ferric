// Copyright 2022 The Ferric AI Project Developers
use syn::parse::{Parse, ParseStream, Result};
use syn::{Error, Expr, Ident, Token};

/// StmtAst is the Abstract Syntax Tree representation of a single dependency statement.
pub struct StmtAst {
    pub var_name: Ident,
    pub type_name: Ident,
    pub dependency: Expr,
}

/// ModelAst is the Abstract Syntax Tree representation of the model.
/// This represents the output of the parse and the analyze phase in a macro pipeline.
pub struct ModelAst {
    pub model_name: Ident,
    pub use_exprs: Vec<Expr>,
    pub stmts: Vec<StmtAst>,
    pub queries: Vec<Ident>,
    pub observes: Vec<Ident>,
}

impl Parse for ModelAst {
    fn parse(input: ParseStream) -> Result<Self> {
        // mod model_name;
        input.parse::<Token![mod]>()?;
        let model_name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let mut stmts = Vec::<StmtAst>::new();
        let mut use_exprs = Vec::<Expr>::new();
        let mut queries = Vec::<Ident>::new();
        let mut observes = Vec::<Ident>::new();

        while !input.is_empty() {
            // let var_name ~ dep_expr;
            if input.peek(Token![let]) {
                input.parse::<Token![let]>()?;
                let var_name: Ident = input.parse()?;
                input.parse::<Token![:]>()?;
                let type_name: Ident = input.parse()?;
                input.parse::<Token![~]>()?;
                let dependency: Expr = input.parse()?;
                input.parse::<Token![;]>()?;
                stmts.push(StmtAst {
                    var_name,
                    type_name,
                    dependency,
                });
            } else if input.peek(Token![use]) {
                input.parse::<Token![use]>()?;
                let use_expr: Expr = input.parse()?;
                input.parse::<Token![;]>()?;
                use_exprs.push(use_expr);
            } else if input.peek(Ident) {
                let keyword: Ident = input.parse()?;
                match keyword.to_string().as_ref() {
                    "observe" => {
                        // observe var_name;
                        let var_name: Ident = input.parse()?;
                        input.parse::<Token![;]>()?;
                        observes.push(var_name);
                    }
                    "query" => {
                        // query var_name;
                        let var_name: Ident = input.parse()?;
                        input.parse::<Token![;]>()?;
                        queries.push(var_name);
                    }
                    _ => {
                        return Err(Error::new(keyword.span(), "expected let | observe | query"));
                    }
                }
            } else {
                return Err(input.error("expected let | observe | query"));
            }
        }
        Ok(ModelAst {
            model_name,
            use_exprs,
            stmts,
            queries,
            observes,
        })
    }
}

#[test]
fn test_parse() {
    use quote::quote;
    use syn::parse2;

    assert!(parse2::<ModelAst>(quote!(
        modu grass;
    ))
    .is_err());
    assert!(parse2::<ModelAst>(quote!(
        mod grass;
        use ferric::distributions::Bernoulli;

        + foo : bool ~ Bernoulli::new( 0.2 );
    ))
    .is_err());
    assert!(parse2::<ModelAst>(quote!(
        mod grass;
        use ferric::distributions::Bernoulli;

        letu rain : bool ~ Bernoulli::new( 0.2 );
    ))
    .is_err());
    assert!(parse2::<ModelAst>(quote!(
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
    .is_ok());
}

#[test]
fn test_analyze() {
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

    let exp_model_name: Ident = parse_quote!(grass);
    assert_eq!(model_ast.model_name, exp_model_name);

    let exp_use_exprs: &Expr = &parse_quote!(ferric::distributions::Bernoulli);
    assert_eq!(model_ast.use_exprs[0], *exp_use_exprs);
    assert_eq!(model_ast.use_exprs.len(), 1);

    let exp_var_name: Ident = parse_quote!(rain);
    let exp_type_name: Ident = parse_quote!(bool);
    let exp_dependency: &Expr = &parse_quote!(Bernoulli::new(0.2));
    assert_eq!(model_ast.stmts[0].var_name, exp_var_name);
    assert_eq!(model_ast.stmts[0].type_name, exp_type_name);
    assert_eq!(model_ast.stmts[0].dependency, *exp_dependency);
    assert_eq!(model_ast.stmts.len(), 3);

    let exp_queryies_0: Ident = parse_quote!(rain);
    let exp_queryies_1: Ident = parse_quote!(sprinkler);
    assert_eq!(model_ast.queries, [exp_queryies_0, exp_queryies_1]);

    let exp_observes_0: Ident = parse_quote!(grass_wet);
    assert_eq!(model_ast.observes, [exp_observes_0]);
}
