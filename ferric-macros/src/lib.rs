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

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn parse_error_produces_compile_error_tokens() {
        // `modu` is not the `mod` keyword, so parsing fails.
        let out = make_model_inner(quote!(modu grass;));
        let s = out.to_string();
        // The resulting TokenStream is a compile_error! invocation.
        assert!(s.contains("compile_error"), "got: {s}");
    }

    #[test]
    fn analyze_error_produces_compile_error_tokens() {
        // Querying a variable that was never declared is an analyze error.
        let out = make_model_inner(quote!(mod grass; query rain;));
        let s = out.to_string();
        assert!(s.contains("compile_error"), "got: {s}");
    }

    #[test]
    fn happy_path_runs_codegen_and_emits_module() {
        // Exercise the success branch through parse -> analyze -> codegen.
        let out = make_model_inner(quote!(
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
        ));
        let s = out.to_string();
        // Sanity-check that codegen produced a `pub mod grass { ... }` item.
        assert!(s.contains("pub mod grass"), "got: {s}");
        assert!(!s.contains("compile_error"), "got: {s}");
    }
}
