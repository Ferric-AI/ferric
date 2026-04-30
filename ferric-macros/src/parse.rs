// Copyright 2022 The Ferric AI Project Developers
use syn::parse::{Parse, ParseStream, Result};
use syn::{Error, Expr, Ident, Token};

/// StmtAst is the Abstract Syntax Tree representation of a single dependency statement.
pub struct StmtAst {
    pub var_ident: Ident,
    pub type_ident: Ident,
    pub dependency: Expr,
}

/// ModelAst is the Abstract Syntax Tree representation of the model.
/// This represents the output of the parse phase in a proc_macro pipeline.
pub struct ModelAst {
    pub model_ident: Ident,
    pub use_exprs: Vec<Expr>,
    pub stmts: Vec<StmtAst>,
    pub queries: Vec<Ident>,
    pub observes: Vec<Ident>,
}

impl Parse for ModelAst {
    fn parse(input: ParseStream) -> Result<Self> {
        // mod model_name;
        input.parse::<Token![mod]>()?;
        let model_ident: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let mut stmts = Vec::<StmtAst>::new();
        let mut use_exprs = Vec::<Expr>::new();
        let mut queries = Vec::<Ident>::new();
        let mut observes = Vec::<Ident>::new();

        while !input.is_empty() {
            // let var_name ~ dep_expr;
            if input.peek(Token![let]) {
                // peek confirmed the token; this parse cannot fail.
                input.parse::<Token![let]>().expect("peek confirmed");
                let var_ident: Ident = input.parse()?;
                input.parse::<Token![:]>()?;
                let type_ident: Ident = input.parse()?;
                input.parse::<Token![~]>()?;
                let dependency: Expr = input.parse()?;
                input.parse::<Token![;]>()?;
                stmts.push(StmtAst {
                    var_ident,
                    type_ident,
                    dependency,
                });
            } else if input.peek(Token![use]) {
                // peek confirmed the token; this parse cannot fail.
                input.parse::<Token![use]>().expect("peek confirmed");
                let use_expr: Expr = input.parse()?;
                input.parse::<Token![;]>()?;
                use_exprs.push(use_expr);
            } else if input.peek(Ident) {
                // peek confirmed an Ident; this parse cannot fail.
                let keyword: Ident = input.parse().expect("peek confirmed");
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
                        return Err(Error::new(
                            keyword.span(),
                            "expected let | use | observe | query",
                        ));
                    }
                }
            } else {
                return Err(input.error("expected let | use | observe | query"));
            }
        }
        Ok(ModelAst {
            model_ident,
            use_exprs,
            stmts,
            queries,
            observes,
        })
    }
}

#[test]
fn test_parse_errors() {
    use quote::quote;
    use syn::parse2;

    assert!(
        parse2::<ModelAst>(quote!(
            modu grass;
        ))
        .is_err()
    );
    assert!(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;

            + foo : bool ~ Bernoulli::new( 0.2 );
        ))
        .is_err()
    );
    assert!(
        parse2::<ModelAst>(quote!(
            mod grass;
            use ferric::distributions::Bernoulli;

            letu rain : bool ~ Bernoulli::new( 0.2 );
        ))
        .is_err()
    );
}

/// Drives parse failures through every `?` early-return in `Parse for ModelAst`,
/// so the failure half of each region is recorded by llvm-cov.
#[test]
fn test_parse_errors_per_token() {
    use quote::quote;
    use syn::parse2;

    // --- mod header ---
    // `mod` keyword missing.
    assert!(parse2::<ModelAst>(quote!(grass;)).is_err());
    // `mod` followed by no model_ident (next token is `;`, not Ident).
    assert!(parse2::<ModelAst>(quote!(mod ;)).is_err());
    // `mod grass` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(mod grass)).is_err());

    // --- let statement ---
    // `let` with no var_ident.
    assert!(parse2::<ModelAst>(quote!(mod m; let ;)).is_err());
    // `let x` with no `:`.
    assert!(parse2::<ModelAst>(quote!(mod m; let x ;)).is_err());
    // `let x :` with no type_ident.
    assert!(parse2::<ModelAst>(quote!(mod m; let x : ;)).is_err());
    // `let x : bool` with no `~`.
    assert!(parse2::<ModelAst>(quote!(mod m; let x : bool ;)).is_err());
    // `let x : bool ~` with no dependency expr.
    assert!(parse2::<ModelAst>(quote!(mod m; let x : bool ~ ;)).is_err());
    // `let x : bool ~ expr` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(mod m; let x : bool ~ Foo)).is_err());

    // --- use statement ---
    // `use` with no expression.
    assert!(parse2::<ModelAst>(quote!(mod m; use ;)).is_err());
    // `use foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(mod m; use foo)).is_err());

    // --- observe statement ---
    // `observe` with no var_name.
    assert!(parse2::<ModelAst>(quote!(mod m; observe ;)).is_err());
    // `observe foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(mod m; observe foo)).is_err());

    // --- query statement ---
    // `query` with no var_name.
    assert!(parse2::<ModelAst>(quote!(mod m; query ;)).is_err());
    // `query foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(mod m; query foo)).is_err());
}

#[test]
fn test_parse_output() {
    use quote::quote;
    use syn::{parse_quote, parse2};

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
    assert_eq!(model_ast.model_ident, exp_model_name);

    let exp_use_exprs: &Expr = &parse_quote!(ferric::distributions::Bernoulli);
    assert_eq!(model_ast.use_exprs[0], *exp_use_exprs);
    assert_eq!(model_ast.use_exprs.len(), 1);

    let exp_var_name: Ident = parse_quote!(rain);
    let exp_type_name: Ident = parse_quote!(bool);
    let exp_dependency: &Expr = &parse_quote!(Bernoulli::new(0.2));
    assert_eq!(model_ast.stmts[0].var_ident, exp_var_name);
    assert_eq!(model_ast.stmts[0].type_ident, exp_type_name);
    assert_eq!(model_ast.stmts[0].dependency, *exp_dependency);
    assert_eq!(model_ast.stmts.len(), 3);

    let exp_queryies_0: Ident = parse_quote!(rain);
    let exp_queryies_1: Ident = parse_quote!(sprinkler);
    assert_eq!(model_ast.queries, [exp_queryies_0, exp_queryies_1]);

    let exp_observes_0: Ident = parse_quote!(grass_wet);
    assert_eq!(model_ast.observes, [exp_observes_0]);
}
