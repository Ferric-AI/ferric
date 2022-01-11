// Copyright 2022 The Ferric AI Project Developers
use syn::parse::{Parse, ParseStream, Result};
use syn::{Error, Expr, Ident, Token};

pub struct Stmt {
    pub var_name: Ident,
    pub type_name: Ident,
    pub dependency: Expr,
}

pub struct ModelStmts {
    pub model_name: Ident,
    pub use_exprs: Vec<Expr>,
    pub stmts: Vec<Stmt>,
    pub queries: Vec<Ident>,
    pub observes: Vec<Ident>,
}

impl Parse for ModelStmts {
    fn parse(input: ParseStream) -> Result<Self> {
        // mod model_name;
        input.parse::<Token![mod]>()?;
        let model_name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let mut stmts = Vec::<Stmt>::new();
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
                stmts.push(Stmt {
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
        Ok(ModelStmts {
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

    assert!(parse2::<ModelStmts>(quote!(
        modu grass;
    ))
    .is_err());
    assert!(parse2::<ModelStmts>(quote!(
        mod grass;
        use ferric::distributions::Bernoulli;

        + foo : bool ~ Bernoulli::new( 0.2 );
    ))
    .is_err());
    assert!(parse2::<ModelStmts>(quote!(
        mod grass;
        use ferric::distributions::Bernoulli;

        letu rain : bool ~ Bernoulli::new( 0.2 );
    ))
    .is_err());
    assert!(parse2::<ModelStmts>(quote!(
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
