// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::Ident;

use crate::analyze::{ModelIR, VariableIR};

pub fn codegen(ir: ModelIR) -> TokenStream {
    let model_ident = ir.model_ident;

    // process all the use statements;
    let mut use_stmts = Vec::<TokenStream>::new();
    for use_expr in ir.use_exprs.iter() {
        let use_stmt = quote! {use #use_expr};
        use_stmts.push(use_stmt);
    }
    // additional use statements needed for code generation
    use_stmts.push(quote! {use ferric::FeOption});

    // for all let <variable name> statements
    let mut var_idents = Vec::<Ident>::new(); // var_<variable name>
    let mut var_type_idents = Vec::<Ident>::new(); // <variable's type name>
    let mut var_eval_idents = Vec::<Ident>::new(); // eval_<variable name>
    let mut var_eval_dist_idents = Vec::<Ident>::new(); // evaldist_<variable name>
    let mut eval_dist_exprs = Vec::<TokenStream>::new(); // expression for evaldist_<variable name>

    // for all query <variable name> statements
    let mut query_idents = Vec::<Ident>::new(); // <variable name>
    let mut query_type_idents = Vec::<Ident>::new(); // <variable's type name>
    let mut query_eval_var_idents = Vec::<Ident>::new(); // eval_<variable name>

    // for all observe <variable name> statements
    let mut obs_idents = Vec::<Ident>::new(); // <variable name>
    let mut obs_var_idents = Vec::<Ident>::new(); // var_<variable name>
    let mut obs_type_idents = Vec::<Ident>::new(); // <variable's type name>
    let mut obs_obs_idents = Vec::<Ident>::new(); // obs_<variable name>
    let mut obs_eval_idents = Vec::<Ident>::new(); // eval_<variable name>

    // process all the variables in the model
    for variable in ir.variables.values() {
        let var_ident = format_ident!("var_{}", &variable.var_ident);
        var_idents.push(var_ident.clone());
        var_type_idents.push(variable.type_ident.clone());
        let eval_var = format_ident!("eval_{}", &variable.var_ident);
        var_eval_idents.push(eval_var.clone());
        let eval_dist_var = format_ident!("evaldist_{}", &variable.var_ident);
        var_eval_dist_idents.push(eval_dist_var.clone());
        let eval_dist_dep = &variable.dependency;
        let modified_eval_dist_dep = replace(quote! {#eval_dist_dep}, &ir.variables);
        eval_dist_exprs.push(modified_eval_dist_dep);
        // if this variable is being queried then we need to store these
        // generated tokens in query-specific lists
        if variable.is_queried {
            query_idents.push(variable.var_ident.clone());
            query_type_idents.push(variable.type_ident.clone());
            query_eval_var_idents.push(eval_var.clone());
        }
        if variable.is_observed {
            obs_idents.push(variable.var_ident.clone());
            obs_type_idents.push(variable.type_ident.clone());
            obs_var_idents.push(var_ident.clone());
            obs_obs_idents.push(format_ident!("obs_{}", &variable.var_ident));
            obs_eval_idents.push(eval_var.clone());
        }
    }

    quote! {
        pub mod #model_ident {
            #(
                #use_stmts;
            )*

            // all the queried variables are fields in the `Sample` struct
            pub struct Sample {
                #(
                    pub #query_idents: #query_type_idents,
                )*
            }

            // all the observed variables are fields in the `Model` struct
            pub struct Model {
                #(
                    pub #obs_idents: #obs_type_idents,
                )*
            }

            impl Model {
                pub fn sample_iter(&self) -> World<rand::rngs::ThreadRng> {
                    World::new(
                        rand::thread_rng(),
                        #(
                            self.#obs_idents,
                        )*
                    )
                }
            }

            pub struct World<R> {
                rng: R,
                #(#var_idents: FeOption<#var_type_idents>, )*
                #(#obs_obs_idents: #obs_type_idents, )*
            }

            impl<R: rand::Rng> Iterator for World<R> {
                type Item = Sample;

                fn next(&mut self) -> Option<Self::Item> {
                    Some(self.sample())
                }
            }

            impl<R: rand::Rng> World<R> {
                pub fn new(rng: R, #(#obs_idents: #obs_type_idents,)*) -> World<R> {
                    World {
                        rng: rng,
                        #(#var_idents: FeOption::Unknown, )*
                        #(#obs_obs_idents: #obs_idents, )*
                    }
                }

                pub fn reset(&mut self) {
                    #(
                        self.#var_idents = FeOption::Unknown;
                    )*
                }

                pub fn sample(&mut self) -> Sample {
                    loop {
                        self.reset();
                        #(
                            if self.#obs_obs_idents != self.#obs_eval_idents() {
                                continue;
                            }
                        )*
                        return Sample {
                            #(
                                #query_idents: self.#query_eval_var_idents(),
                            )*
                        };
                    }
                }

                #(
                pub fn #var_eval_idents(&mut self) -> #var_type_idents {
                    if self.#var_idents.is_unknown() {
                        let dist = self.#var_eval_dist_idents();
                        self.#var_idents = FeOption::Known(dist.sample(&mut self.rng));
                    }
                    // TODO: handle the case when the value is Null
                    self.#var_idents.unwrap()
                }

                pub fn #var_eval_dist_idents(&mut self) -> Box<dyn ferric::distributions::Distribution<R, Domain=#var_type_idents>> {
                    // TODO: handle errors in constructing the distribution object
                    let dist = #eval_dist_exprs;
                    Box::new(dist.unwrap())
                }
                )*
            }
        }
    }
}

// replace all occurrences of `var_name` in the dependency expression with `self.eval_var_name()`
fn replace(dep_tokens: TokenStream, variables: &HashMap<String, VariableIR>) -> TokenStream {
    let dep_it = dep_tokens.into_iter();

    dep_it
        .map(|tt| match tt {
            TokenTree::Ident(ref i) => {
                let i_name = &i.to_string();
                if variables.contains_key(i_name) {
                    let var_ident = Ident::new(&format!("eval_{}", i_name), i.span());
                    let stream = quote! {self.#var_ident()};
                    let mut group = Group::new(Delimiter::None, stream);
                    group.set_span(i.span());
                    TokenTree::Group(group)
                } else {
                    tt
                }
            }
            TokenTree::Group(ref g) => {
                let stream = replace(g.stream(), variables);
                let mut group = Group::new(g.delimiter(), stream);
                group.set_span(g.span());
                TokenTree::Group(group)
            }
            other => other,
        })
        .collect()
}

#[test]
fn output_is_module_item() {
    use proc_macro2::Span;
    use syn::{parse2, parse_quote, ItemMod};
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("grass"), Span::call_site()),
        use_exprs: vec![parse_quote!(ferric::distributions::Bernoulli)],
        variables: HashMap::from([
            (
                String::from("rain"),
                VariableIR {
                    var_ident: Ident::new(&String::from("rain"), Span::call_site()),
                    type_ident: Ident::new(&String::from("bool"), Span::call_site()),
                    dependency: parse_quote!(Bernoulli::new(0.2)),
                    is_queried: true,
                    is_observed: false,
                },
            ),
            (
                String::from("sprinkler"),
                VariableIR {
                    var_ident: Ident::new(&String::from("sprinkler"), Span::call_site()),
                    type_ident: Ident::new(&String::from("bool"), Span::call_site()),
                    dependency: parse_quote!(if rain {
                        Bernoulli::new(0.01)
                    } else {
                        Bernoulli::new(0.4)
                    }),
                    is_queried: false,
                    is_observed: true,
                },
            ),
        ]),
    };
    let rust = codegen(ir);

    assert!(parse2::<ItemMod>(rust).is_ok());
}
