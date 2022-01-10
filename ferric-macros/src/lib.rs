// Copyright 2022 The Ferric AI Project Developers

use proc_macro2::Ident as Ident2;
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{parse2, Ident};

mod parse;
use crate::parse::ModelAst;

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
    let ast = parse2::<ModelAst>(input);
    let ast = match ast {
        Ok(data) => data,
        Err(err) => {
            return err.to_compile_error();
        }
    };
    let model_name = ast.model_name;

    // process all the use statements;
    let mut use_names = Vec::<TokenStream>::new();
    for use_expr in ast.use_exprs.iter() {
        let use_name = quote! {use #use_expr};
        use_names.push(use_name);
    }

    // for all let <variable name> statements
    let mut var_names_set = HashSet::<String>::new(); // var_<variable name>
    let mut var_names = Vec::<Ident>::new(); // <variable name>
    let mut var_type_names = Vec::<Ident>::new(); // <variable's type name>
    let mut var_init_names = Vec::<Ident>::new(); // initialized_<variable name>
    let mut var_eval_names = Vec::<Ident>::new(); // eval_<variable name>
    let mut var_eval_dist_names = Vec::<Ident>::new(); // evaldist_<variable name>
                                                       // for all query <variable name> statements
    let mut query_names = Vec::<Ident>::new(); // <variable name>
    let mut query_type_names = Vec::<Ident>::new(); // <variable's type name>
    let mut query_eval_var_names = Vec::<Ident>::new(); // eval_<variable name>
                                                        // for all observe <variable name> statements
    let mut obs_names = Vec::<Ident>::new(); // <variable name>
    let mut obs_var_names = Vec::<Ident>::new(); // var_<variable name>
    let mut obs_type_names = Vec::<Ident>::new(); // <variable's type name>
    let mut obs_obs_names = Vec::<Ident>::new(); // obs_<variable name>
    let mut obs_eval_names = Vec::<Ident>::new(); // eval_<variable name>
                                                  // process all the let statements
    for stmt in ast.stmts.iter() {
        var_names_set.insert(stmt.var_name.to_string());
        let var_name = format_ident!("var_{}", &stmt.var_name);
        var_names.push(var_name.clone());
        var_type_names.push(stmt.type_name.clone());
        let init_var = format_ident!("initialized_{}", &stmt.var_name);
        var_init_names.push(init_var.clone());
        let eval_var = format_ident!("eval_{}", &stmt.var_name);
        var_eval_names.push(eval_var.clone());
        let eval_dist_var = format_ident!("evaldist_{}", &stmt.var_name);
        var_eval_dist_names.push(eval_dist_var.clone());
        // if this variable is being queried then we need to store these
        // generate tokens in query-specific lists
        if ast.queries.contains(&stmt.var_name) {
            query_names.push(stmt.var_name.clone());
            query_type_names.push(stmt.type_name.clone());
            query_eval_var_names.push(eval_var.clone());
        }
        if ast.observes.contains(&stmt.var_name) {
            obs_names.push(stmt.var_name.clone());
            obs_type_names.push(stmt.type_name.clone());
            obs_var_names.push(var_name.clone());
            obs_obs_names.push(format_ident!("obs_{}", &stmt.var_name));
            obs_eval_names.push(eval_var.clone());
        }
    }

    // TODO: check that a variable doesn't have a duplicate definition
    // TODO: check that query_name is identical to ast.queries

    // once we have all the variable names then we can replace them
    let mut eval_dist_exprs = Vec::<TokenStream>::new();
    for stmt in ast.stmts.iter() {
        let eval_dist_dep = &stmt.dependency;
        let modified_eval_dist_dep = replace(quote! {#eval_dist_dep}, &var_names_set);
        eval_dist_exprs.push(modified_eval_dist_dep);
    }

    quote! {
        pub mod #model_name {
            #(
                #use_names;
            )*

            pub struct Sample {
                #(pub #query_names: #query_type_names, )*
            }

            pub struct Model {
                #(
                    pub #obs_names: #obs_type_names,
                )*
            }

            impl Model {
                pub fn sample_iter(&self) -> World<rand::rngs::ThreadRng> {
                    World::new(
                        rand::thread_rng(),
                        #(
                            self.#obs_names,
                        )*
                    )
                }
            }

            pub struct World<R> {
                rng: R,
                #(#var_names: #var_type_names, )*
                #(#var_init_names: bool, )*
                #(#obs_obs_names: #obs_type_names, )*
            }

            impl<R: rand::Rng> Iterator for World<R> {
                type Item = Sample;

                fn next(&mut self) -> Option<Self::Item> {
                    Some(self.sample())
                }
            }

            impl<R: rand::Rng> World<R> {
                pub fn new(rng: R, #(#obs_names: #obs_type_names,)*) -> World<R> {
                    World {
                        rng: rng,
                        #(#var_names: Default::default(), )*
                        #(#var_init_names: false, )*
                        #(#obs_obs_names: #obs_names, )*
                    }
                }

                pub fn reset(&mut self) {
                    #(
                        self.#var_init_names = false;
                    )*
                }

                pub fn sample(&mut self) -> Sample {
                    loop {
                        self.reset();
                        #(
                            if self.#obs_obs_names != self.#obs_eval_names() {
                                continue;
                            }
                        )*
                        return Sample {
                            #(
                                #query_names: self.#query_eval_var_names(),
                            )*
                        };
                    }
                }

                #(
                pub fn #var_eval_names(&mut self) -> #var_type_names {
                    if !self.#var_init_names {
                        let dist = self.#var_eval_dist_names();
                        self.#var_names = dist.sample(&mut self.rng);
                        self.#var_init_names = true;
                    }
                    self.#var_names
                }

                pub fn #var_eval_dist_names(&mut self) -> Box<dyn ferric::distributions::Distribution<R, Domain=#var_type_names>> {
                    // TODO: handle errors in constructing the distribution object
                    let dist = #eval_dist_exprs;
                    //println!("running {}", #var_eval_dist_names);
                    Box::new(dist.unwrap())
                }
                )*
            }
        }
    }
}

// replace all occurrences of `var_name` in the dependency expression with `self.eval_var_name()`
fn replace(dep_tokens: TokenStream, var_names_set: &HashSet<String>) -> TokenStream {
    let dep_it = dep_tokens.into_iter();

    dep_it
        .map(|tt| match tt {
            TokenTree::Ident(ref i) => {
                let i_name = &i.to_string();
                if var_names_set.contains(i_name) {
                    let var_name = Ident2::new(&format!("eval_{}", i_name), i.span());
                    let stream = quote! {self.#var_name()};
                    let mut group = Group::new(Delimiter::None, stream);
                    group.set_span(i.span());
                    TokenTree::Group(group)
                } else {
                    tt
                }
            }
            TokenTree::Group(ref g) => {
                let stream = replace(g.stream(), var_names_set);
                let mut group = Group::new(g.delimiter(), stream);
                group.set_span(g.span());
                TokenTree::Group(group)
            }
            other => other,
        })
        .collect()
}

#[test]
fn test_make_model() {
    make_model_inner(quote!(modu grass;));
}
