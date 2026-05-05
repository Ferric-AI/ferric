// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::{TokenStream, TokenTree};
use syn::parse::{Parse, ParseStream, Result};
use syn::spanned::Spanned;
use syn::{Error, Expr, Ident, Token, Type, bracketed};

/// A zero-based index range used by an indexed random variable.
///
/// Ferric accepts ranges of the form `idx of n`, where `idx` is the name made
/// available inside the dependency expression and `n` is a previously declared
/// constant or random variable. The index takes values `0` through `n - 1`.
#[derive(Clone)]
pub struct IndexRangeAst {
    pub index_ident: Ident,
    pub upper_ident: Ident,
}

/// Constant declaration supplied when a model is instantiated.
pub struct ConstAst {
    pub const_ident: Ident,
    pub type_ident: Type,
    pub order: usize,
}

/// StmtAst is the Abstract Syntax Tree representation of a single dependency statement.
pub struct StmtAst {
    pub var_ident: Ident,
    pub indices: Vec<IndexRangeAst>,
    pub type_ident: Type,
    pub dependency: TokenStream,
    pub max_expr: Option<Expr>,
    pub max_const_ident: Option<Ident>,
    /// `true` if defined with `~` (distribution), `false` if defined with `=` (deterministic expr).
    pub is_stochastic: bool,
    pub order: usize,
}

/// ModelAst is the Abstract Syntax Tree representation of the model.
/// This represents the output of the parse phase in a proc_macro pipeline.
pub struct ModelAst {
    pub model_ident: Ident,
    pub consts: Vec<ConstAst>,
    pub use_exprs: Vec<Expr>,
    pub stmts: Vec<StmtAst>,
    pub queries: Vec<Ident>,
    pub observes: Vec<Ident>,
}

impl Parse for ModelAst {
    fn parse(input: ParseStream) -> Result<Self> {
        // name model_name;
        let keyword: Ident = input.parse()?;
        if keyword != "name" {
            return Err(Error::new(keyword.span(), "expected `name`"));
        }
        let model_ident: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let mut stmts = Vec::<StmtAst>::new();
        let mut consts = Vec::<ConstAst>::new();
        let mut use_exprs = Vec::<Expr>::new();
        let mut queries = Vec::<Ident>::new();
        let mut observes = Vec::<Ident>::new();
        let mut order = 0usize;

        while !input.is_empty() {
            if input.peek(Token![const]) {
                input.parse::<Token![const]>().expect("peek confirmed");
                let const_ident: Ident = input.parse()?;
                input.parse::<Token![:]>()?;
                let type_ident: Type = input.parse()?;
                input.parse::<Token![;]>()?;
                consts.push(ConstAst {
                    const_ident,
                    type_ident,
                    order,
                });
                order += 1;
            } else if input.peek(Token![let]) {
                // let var_name[row of n, col of m] : Type ~ dep_expr;
                // peek confirmed the token; this parse cannot fail.
                input.parse::<Token![let]>().expect("peek confirmed");
                let var_ident: Ident = input.parse()?;
                let indices = if input.peek(syn::token::Bracket) {
                    parse_indices(input)?
                } else {
                    Vec::new()
                };
                input.parse::<Token![:]>()?;
                let type_ident: Type = input.parse()?;
                let is_stochastic = if input.peek(Token![~]) {
                    input.parse::<Token![~]>().expect("peek confirmed");
                    true
                } else if input.peek(Token![=]) {
                    input.parse::<Token![=]>().expect("peek confirmed");
                    false
                } else {
                    return Err(input.error("expected `~` or `=`"));
                };
                let dependency = parse_dependency(input)?;
                let (max_expr, max_const_ident) = parse_optional_max(input)?;
                input.parse::<Token![;]>()?;
                stmts.push(StmtAst {
                    var_ident,
                    indices,
                    type_ident,
                    dependency,
                    max_expr,
                    max_const_ident,
                    is_stochastic,
                    order,
                });
                order += 1;
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
            consts,
            use_exprs,
            stmts,
            queries,
            observes,
        })
    }
}

fn parse_indices(input: ParseStream) -> Result<Vec<IndexRangeAst>> {
    let content;
    bracketed!(content in input);
    let mut indices = Vec::new();
    while !content.is_empty() {
        let index_ident: Ident = content.parse()?;
        let keyword: Ident = content.parse()?;
        if keyword != "of" {
            return Err(Error::new(keyword.span(), "expected `of`"));
        }
        let upper_ident: Ident = content.parse()?;
        indices.push(IndexRangeAst {
            index_ident,
            upper_ident,
        });
        if content.is_empty() {
            break;
        }
        content.parse::<Token![,]>()?;
    }
    Ok(indices)
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn parse_optional_max(input: ParseStream) -> Result<(Option<Expr>, Option<Ident>)> {
    if !input.peek(Ident) {
        return Ok((None, None));
    }

    let fork = input.fork();
    let keyword: Ident = fork.parse().expect("peek confirmed");
    if keyword != "max" {
        return Ok((None, None));
    }

    let _: Ident = input.parse().expect("peek confirmed");
    let max_expr: Expr = input.parse()?;
    let max_const_ident = match &max_expr {
        Expr::Lit(_) => None,
        Expr::Path(path) => match path.path.get_ident() {
            Some(ident) => Some(ident.clone()),
            None => {
                return Err(Error::new(
                    max_expr.span(),
                    "`max ...` must be a literal or a constant identifier",
                ));
            }
        },
        _ => {
            return Err(Error::new(
                max_expr.span(),
                "`max ...` must be a literal or a constant identifier",
            ));
        }
    };
    Ok((Some(max_expr), max_const_ident))
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn parse_dependency(input: ParseStream) -> Result<TokenStream> {
    let mut tokens = TokenStream::new();
    while !input.is_empty() {
        if input.peek(Token![;]) {
            break;
        }
        if input.peek(Ident) {
            let fork = input.fork();
            let keyword: Ident = fork.parse().expect("peek confirmed");
            if keyword == "max" {
                break;
            }
        }
        let tt: TokenTree = input.parse()?;
        tokens.extend(TokenStream::from(tt));
    }
    if tokens.is_empty() {
        return Err(input.error("expected dependency expression"));
    }
    Ok(tokens)
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
            name grass;
            use ferric::distributions::Bernoulli;

            + foo : bool ~ Bernoulli::new( 0.2 );
        ))
        .is_err()
    );
    assert!(
        parse2::<ModelAst>(quote!(
            name grass;
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

    // --- name header ---
    // `name` keyword missing.
    assert!(parse2::<ModelAst>(quote!(grass;)).is_err());
    // `mod` is no longer accepted as a model header.
    assert!(
        parse2::<ModelAst>(quote!(
            mod grass;
        ))
        .is_err()
    );
    // `name` followed by no model_ident (next token is `;`, not Ident).
    assert!(parse2::<ModelAst>(quote!(name ;)).is_err());
    // `name grass` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name grass)).is_err());

    // --- const statement ---
    // `const` with no name.
    assert!(parse2::<ModelAst>(quote!(name m; const ;)).is_err());
    // `const n` with no `:`.
    assert!(parse2::<ModelAst>(quote!(name m; const n ;)).is_err());
    // `const n :` with no type.
    assert!(parse2::<ModelAst>(quote!(name m; const n : ;)).is_err());
    // `const n : u64` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; const n : u64)).is_err());

    // --- let statement ---
    // `let` with no var_ident.
    assert!(parse2::<ModelAst>(quote!(name m; let ;)).is_err());
    // `let x` with no `:`.
    assert!(parse2::<ModelAst>(quote!(name m; let x ;)).is_err());
    // `let x :` with no type.
    assert!(parse2::<ModelAst>(quote!(name m; let x : ;)).is_err());
    // `let x : bool` with neither `~` nor `=`.
    assert!(parse2::<ModelAst>(quote!(name m; let x : bool ;)).is_err());
    // `let x : bool ~` with no dependency expr.
    assert!(parse2::<ModelAst>(quote!(name m; let x : bool ~ ;)).is_err());
    // `let x : bool ~ expr` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; let x : bool ~ Foo)).is_err());
    // `let x : bool ~ expr max` with no max expression.
    assert!(parse2::<ModelAst>(quote!(name m; let x : bool ~ Foo max ;)).is_err());
    // `let x : bool ~ expr unexpected` leaves the unexpected ident for `;` parsing.
    // `max` accepts a literal or single identifier, not an expression.
    assert!(parse2::<ModelAst>(quote!(name m; let x : u64 ~ Foo max limit + 1;)).is_err());
    // `max` accepts a literal or single identifier, not a path.
    assert!(parse2::<ModelAst>(quote!(name m; let x : u64 ~ Foo max foo::bar;)).is_err());
    // `let x : bool = expr` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; let x : bool = true)).is_err());
    // Indexed variables must use named zero-based ranges.
    assert!(parse2::<ModelAst>(quote!(name m; let x[0:n] : bool ~ Foo;)).is_err());
    // Missing index name inside brackets.
    assert!(parse2::<ModelAst>(quote!(name m; let x[of n] : bool ~ Foo;)).is_err());
    // Missing `of` inside an index range.
    assert!(parse2::<ModelAst>(quote!(name m; let x[i n] : bool ~ Foo;)).is_err());
    // Missing upper-bound identifier.
    assert!(parse2::<ModelAst>(quote!(name m; let x[i of] : bool ~ Foo;)).is_err());
    // Missing `of` inside an index range.
    assert!(parse2::<ModelAst>(quote!(name m; let x[i] : bool ~ Foo;)).is_err());
    // Missing upper-bound identifier.
    assert!(parse2::<ModelAst>(quote!(name m; let x[i of] : bool ~ Foo;)).is_err());
    // Missing comma between index ranges.
    assert!(parse2::<ModelAst>(quote!(name m; let x[i of n j of m] : bool ~ Foo;)).is_err());

    // --- use statement ---
    // `use` with no expression.
    assert!(parse2::<ModelAst>(quote!(name m; use ;)).is_err());
    // `use foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; use foo)).is_err());

    // --- observe statement ---
    // `observe` with no var_name.
    assert!(parse2::<ModelAst>(quote!(name m; observe ;)).is_err());
    // `observe foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; observe foo)).is_err());

    // --- query statement ---
    // `query` with no var_name.
    assert!(parse2::<ModelAst>(quote!(name m; query ;)).is_err());
    // `query foo` with no trailing `;`.
    assert!(parse2::<ModelAst>(quote!(name m; query foo)).is_err());
}

#[test]
fn test_parse_output() {
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

    let exp_model_name: Ident = parse_quote!(grass);
    assert_eq!(model_ast.model_ident, exp_model_name);

    let exp_use_exprs: &Expr = &parse_quote!(ferric::distributions::Bernoulli);
    assert_eq!(model_ast.use_exprs[0], *exp_use_exprs);
    assert_eq!(model_ast.use_exprs.len(), 1);

    let exp_var_name: Ident = parse_quote!(rain);
    let exp_type_name: Type = parse_quote!(bool);
    let exp_dependency: TokenStream = parse_quote!(Bernoulli::new(0.2));
    assert_eq!(model_ast.stmts[0].var_ident, exp_var_name);
    assert_eq!(model_ast.stmts[0].type_ident, exp_type_name);
    assert_eq!(
        model_ast.stmts[0].dependency.to_string(),
        exp_dependency.to_string()
    );
    assert_eq!(model_ast.stmts.len(), 3);

    let exp_queryies_0: Ident = parse_quote!(rain);
    let exp_queryies_1: Ident = parse_quote!(sprinkler);
    assert_eq!(model_ast.queries, [exp_queryies_0, exp_queryies_1]);

    let exp_observes_0: Ident = parse_quote!(grass_wet);
    assert_eq!(model_ast.observes, [exp_observes_0]);

    assert!(model_ast.stmts[0].is_stochastic);
    assert!(model_ast.stmts[1].is_stochastic);
    assert!(model_ast.stmts[2].is_stochastic);
}

#[test]
fn test_parse_const_and_indexed_stmt() {
    use quote::quote;
    use syn::{parse_quote, parse2};

    let model_ast = parse2::<ModelAst>(quote!(
        name indexed;
        const n : u64;
        const m : u64;
        let grid[row of n, col of m] : bool ~ Bernoulli::new(0.5);
        query grid;
    ))
    .unwrap();

    let exp_n: Ident = parse_quote!(n);
    let exp_m: Ident = parse_quote!(m);
    let exp_row: Ident = parse_quote!(row);
    let exp_col: Ident = parse_quote!(col);
    let exp_grid: Ident = parse_quote!(grid);
    assert_eq!(model_ast.consts.len(), 2);
    assert_eq!(model_ast.consts[0].const_ident, exp_n);
    assert_eq!(model_ast.consts[1].const_ident, exp_m);
    assert_eq!(model_ast.stmts[0].var_ident, exp_grid);
    assert_eq!(model_ast.stmts[0].indices.len(), 2);
    assert_eq!(model_ast.stmts[0].indices[0].index_ident, exp_row);
    assert_eq!(model_ast.stmts[0].indices[0].upper_ident, exp_n);
    assert_eq!(model_ast.stmts[0].indices[1].index_ident, exp_col);
    assert_eq!(model_ast.stmts[0].indices[1].upper_ident, exp_m);
    assert!(model_ast.stmts[0].max_expr.is_none());
}

#[test]
fn test_parse_max_stmt() {
    use quote::quote;
    use syn::{parse_quote, parse2};

    let model_ast = parse2::<ModelAst>(quote!(
        name indexed;
        const max_n : u64;
        let n : u64 ~ Poisson::new(3.0) max max_n;
        query n;
    ))
    .unwrap();

    let exp_max: Expr = parse_quote!(max_n);
    assert_eq!(model_ast.stmts[0].max_expr, Some(exp_max));
}

#[test]
fn test_parse_deterministic_stmt() {
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

    assert!(model_ast.stmts[0].is_stochastic);
    assert!(!model_ast.stmts[1].is_stochastic);

    let exp_dep: TokenStream = parse_quote!(2u8 * x as u8);
    assert_eq!(
        model_ast.stmts[1].dependency.to_string(),
        exp_dep.to_string()
    );
}

#[test]
fn test_parse_vec_type() {
    use quote::quote;
    use syn::{parse_quote, parse2};

    let model_ast = parse2::<ModelAst>(quote!(
        name dirichlet_model;
        use ferric::distributions::Dirichlet;

        let theta : Vec<f64> ~ Dirichlet::new(vec![1.0, 1.0, 1.0]);

        query theta;
    ))
    .unwrap();

    let exp_type: Type = parse_quote!(Vec<f64>);
    assert_eq!(model_ast.stmts[0].type_ident, exp_type);
}
