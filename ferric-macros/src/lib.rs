// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::TokenStream;
use syn::parse2;

mod parse;
use crate::parse::ModelAst;
mod analyze;
use crate::analyze::analyze;
mod codegen;
use crate::codegen::codegen;

//
// Parse statements such as the following
//   let rain ~ Bernoulli(0.2);
// to build a Ferric model.
//
#[proc_macro]
pub fn make_model(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    make_model_inner(TokenStream::from(input)).into()
}

fn make_model_inner(input: TokenStream) -> TokenStream {
    // parse the token stream into an AST (abstract syntax tree)
    let ast = parse2::<ModelAst>(input);
    let ast = match ast {
        Ok(data) => data,
        Err(err) => {
            return err.to_compile_error();
        }
    };
    // analyze the AST and produce an IR (intermediate representation)
    let ir = analyze(ast);
    let ir = match ir {
        Ok(data) => data,
        Err(err) => {
            return err.to_compile_error();
        }
    };
    codegen(ir)
}

#[test]
fn test_parse_error() {
    use quote::quote;
    make_model_inner(quote!(modu grass;));
}

#[test]
fn test_analyze_error() {
    use quote::quote;
    make_model_inner(quote!(mod grass; query rain;));
}
