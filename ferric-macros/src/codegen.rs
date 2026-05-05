// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Ident, Type, parse_quote};

use crate::analyze::{ConstantIR, ModelIR, VariableIR};

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

    let mut const_model_fields = Vec::<TokenStream>::new();
    let mut const_world_fields = Vec::<TokenStream>::new();
    let mut const_new_params = Vec::<TokenStream>::new();
    let mut const_new_inits = Vec::<TokenStream>::new();
    let mut const_model_args = Vec::<TokenStream>::new();

    for konst in ir.consts.values() {
        let const_ident = &konst.const_ident;
        let field_ident = const_field_ident(konst);
        let ty = &konst.type_ident;
        const_model_fields.push(quote! { pub #const_ident: #ty, });
        const_world_fields.push(quote! { #field_ident: #ty, });
        const_new_params.push(quote! { #const_ident: #ty, });
        const_new_inits.push(quote! { #field_ident: #const_ident, });
        const_model_args.push(quote! { self.#const_ident.clone(), });
    }

    let mut var_fields = Vec::<TokenStream>::new();
    let mut var_inits = Vec::<TokenStream>::new();
    let mut reset_stmts = Vec::<TokenStream>::new();
    let mut reset_for_weighted_stmts = Vec::<TokenStream>::new();
    let mut bound_log_cum_fields = Vec::<TokenStream>::new();
    let mut bound_log_cum_inits = Vec::<TokenStream>::new();
    let mut bound_log_cum_reset_stmts = Vec::<TokenStream>::new();
    let mut weighted_pin_stmts = Vec::<TokenStream>::new();
    let mut eval_methods = Vec::<TokenStream>::new();
    let mut query_fields = Vec::<TokenStream>::new();
    let mut query_sample_fields = Vec::<TokenStream>::new();
    let mut obs_model_fields = Vec::<TokenStream>::new();
    let mut obs_world_fields = Vec::<TokenStream>::new();
    let mut obs_new_params = Vec::<TokenStream>::new();
    let mut obs_new_inits = Vec::<TokenStream>::new();
    let mut obs_model_args = Vec::<TokenStream>::new();
    let mut rejection_checks = Vec::<TokenStream>::new();
    let mut weighted_log_terms = Vec::<TokenStream>::new();

    // process all the variables in the model
    for variable in ir.variables.values() {
        let var_ident = format_ident!("var_{}", &variable.var_ident);
        let eval_var = format_ident!("eval_{}", &variable.var_ident);
        let value_type = value_type(&variable.type_ident, variable.indices.len());
        if variable.indices.is_empty() {
            var_fields.push(quote! { #var_ident: FeOption<#value_type>, });
        } else {
            let cache_type = cache_type(&variable.type_ident, variable.indices.len());
            var_fields.push(quote! { #var_ident: FeOption<#cache_type>, });
        }
        var_inits.push(quote! { #var_ident: FeOption::Unknown, });
        reset_stmts.push(quote! { self.#var_ident = FeOption::Unknown; });

        if variable.is_stochastic && variable.max_expr.is_some() {
            let field_ident = bound_log_cum_field_ident(variable);
            let field_type = crate::codegen::value_type(&parse_quote!(f64), variable.indices.len());
            bound_log_cum_fields.push(quote! { #field_ident: FeOption<#field_type>, });
            bound_log_cum_inits.push(quote! { #field_ident: FeOption::Unknown, });
            bound_log_cum_reset_stmts.push(quote! { self.#field_ident = FeOption::Unknown; });
        }

        eval_methods.push(eval_methods_for(variable, &ir.variables, &ir.consts));
        eval_methods.push(bound_log_cum_method_for(
            variable,
            &ir.variables,
            &ir.consts,
        ));

        if variable.is_queried {
            let query_ident = &variable.var_ident;
            query_fields.push(quote! { pub #query_ident: #value_type, });
            query_sample_fields.push(quote! { #query_ident: self.#eval_var(), });
        }

        if variable.is_observed {
            let obs_ident = &variable.var_ident;
            let obs_field = format_ident!("obs_{}", &variable.var_ident);
            let obs_type = observed_type(&variable.type_ident, variable.indices.len());
            obs_model_fields.push(quote! { pub #obs_ident: #obs_type, });
            obs_world_fields.push(quote! { #obs_field: #obs_type, });
            obs_new_params.push(quote! { #obs_ident: #obs_type, });
            obs_new_inits.push(quote! { #obs_field: #obs_ident, });
            obs_model_args.push(quote! { self.#obs_ident.clone(), });
            if variable.indices.is_empty() {
                rejection_checks.push(quote! {
                    {
                        let sampled = self.#eval_var();
                        if self.#obs_field != sampled {
                            continue;
                        }
                    }
                });
            } else {
                rejection_checks.push(quote! {
                    {
                        let sampled = self.#eval_var();
                        if !ferric::MaskedEq::masked_eq(&self.#obs_field, &sampled) {
                            continue;
                        }
                    }
                });
            }

            if variable.is_stochastic && variable.indices.is_empty() {
                let eval_dist_var = format_ident!("evaldist_{}", &variable.var_ident);
                let log_prob_expr = weighted_log_prob_expr(
                    quote! { dist },
                    quote! { &self.#obs_field },
                    variable,
                    &ir.variables,
                    &ir.consts,
                    Some(quote! { bound_log_cum }),
                );
                weighted_pin_stmts.push(quote! {
                    self.#var_ident = FeOption::Known(self.#obs_field.clone());
                });
                let bound_log_cum = if variable.max_expr.is_some() {
                    let eval_bound = bound_log_cum_eval_ident(variable);
                    quote! { let bound_log_cum = self.#eval_bound(); }
                } else {
                    quote! {}
                };
                weighted_log_terms.push(quote! {
                    {
                        let dist = self.#eval_dist_var();
                        #bound_log_cum
                        log_weight += #log_prob_expr;
                    }
                });
            } else if variable.is_stochastic {
                reset_for_weighted_stmts.push(quote! {
                    self.#var_ident = FeOption::Unknown;
                });
                weighted_log_terms.push(array_weighted_log_term(
                    variable,
                    &ir.variables,
                    &ir.consts,
                ));
            } else {
                // Deterministic observed: reset to Unknown in reset_for_weighted so it gets
                // re-evaluated from the freshly sampled stochastic variables.
                reset_for_weighted_stmts.push(quote! {
                    self.#var_ident = FeOption::Unknown;
                });
            }
        } else {
            reset_for_weighted_stmts.push(quote! {
                self.#var_ident = FeOption::Unknown;
            });
        }
    }

    // Weighted sampling is only valid when every observed variable is stochastic.
    // When a deterministic variable is observed we cannot evaluate its log-likelihood,
    // so we omit the weighted-sampling infrastructure entirely.
    let has_det_observed = ir
        .variables
        .values()
        .any(|v| v.is_observed && !v.is_stochastic);

    let weighted_structs = if !has_det_observed {
        quote! {
            /// A sample returned by self-normalised importance sampling via
            /// [`Model::weighted_sample_iter`].
            ///
            /// The queried variables live in the nested `sample` field so
            /// that `log_weight` can never collide with a user-defined
            /// random variable name.  Use [`ferric::weighted_mean`] and
            /// [`ferric::weighted_std`] to compute posterior statistics.
            ///
            /// # Example access pattern
            ///
            /// ```text
            /// for ws in model.weighted_sample_iter().take(n) {
            ///     vals.push(ws.sample.my_var as u8 as f64);
            ///     log_weights.push(ws.log_weight);
            /// }
            /// ```
            pub struct WeightedSample {
                /// Sum of log-likelihoods of all observations under the
                /// latent variables drawn in this sample.  Produced by
                /// $\sum_i \log p(\text{obs}_i \mid \text{latents})$.
                pub log_weight: f64,
                /// The queried variable values for this sample.
                pub sample: Sample,
            }

            /// Iterator adaptor over [`World`] that yields [`WeightedSample`]s
            /// from self-normalised importance sampling.
            ///
            /// Obtain one via [`Model::weighted_sample_iter`].
            pub struct WeightedWorld<R>(World<R>);

            impl<R: rand::Rng> Iterator for WeightedWorld<R> {
                type Item = WeightedSample;

                fn next(&mut self) -> Option<Self::Item> {
                    Some(self.0.weighted_sample())
                }
            }
        }
    } else {
        quote! {}
    };

    let weighted_sample_iter_method = if !has_det_observed {
        quote! {
            /// Returns an iterator of importance-weighted samples using
            /// self-normalised importance sampling (SNIS).
            ///
            /// Each call to [`Iterator::next`] draws the latent variables
            /// from their priors and sets
            ///
            /// ```text
            /// log_weight = Σ log p(obs_i | latent variables)
            /// ```
            ///
            /// Collect the `log_weight` values alongside the queried
            /// fields and pass them to [`ferric::weighted_mean`] or
            /// [`ferric::weighted_std`] to obtain posterior estimates.
            ///
            /// # When to use
            ///
            /// Valid for **all** models where every observed variable is
            /// stochastic (has a distribution).  Also correct (though less
            /// sample-efficient than rejection sampling) for purely discrete
            /// models.
            pub fn weighted_sample_iter(&self) -> WeightedWorld<rand::rngs::ThreadRng> {
                WeightedWorld(World::new(
                    rand::thread_rng(),
                    #(#const_model_args)*
                    #(#obs_model_args)*
                ))
            }
        }
    } else {
        quote! {}
    };

    let weighted_sample_method = if !has_det_observed {
        quote! {
            /// Draw one importance-weighted sample via self-normalised
            /// importance sampling.
            ///
            /// Latent variables are sampled from their priors; the
            /// `log_weight` field is set to
            ///
            /// ```text
            /// log_weight = Σ log p(obs_i | latent variables)
            /// ```
            ///
            /// Valid for discrete and continuous observations alike.
            pub fn weighted_sample(&mut self) -> WeightedSample {
                self.reset_for_weighted();
                let mut log_weight = 0.0f64;
                #(#weighted_log_terms)*
                WeightedSample {
                    log_weight,
                    sample: Sample {
                        #(#query_sample_fields)*
                    },
                }
            }
        }
    } else {
        quote! {}
    };

    let active_eval_method = active_eval_method_for(&ir.variables);

    quote! {
        pub mod #model_ident {
            #(
                #use_stmts;
            )*

            /// A sample returned by rejection sampling via [`Model::sample_iter`].
            ///
            /// Every observed variable matched its observed value exactly, so
            /// all queried fields are drawn from the exact posterior.
            pub struct Sample {
                #(#query_fields)*
            }

            #weighted_structs

            /// The observed data for the model.
            pub struct Model {
                #(#const_model_fields)*
                #(#obs_model_fields)*
            }

            impl Model {
                /// Returns an iterator of exact posterior samples via
                /// rejection sampling.
                ///
                /// Each call to [`Iterator::next`] loops internally until a
                /// prior sample is consistent with every observation, then
                /// returns that sample.
                ///
                /// # When to use
                ///
                /// Only valid when **all** observed variables are discrete.
                /// Conditioning on a continuous observed value has probability
                /// zero and this iterator will loop forever.  Use
                /// [`Model::weighted_sample_iter`] for models with continuous
                /// observations.
                pub fn sample_iter(&self) -> World<rand::rngs::ThreadRng> {
                    World::new(
                        rand::thread_rng(),
                        #(#const_model_args)*
                        #(#obs_model_args)*
                    )
                }

                #weighted_sample_iter_method
            }

            pub struct World<R> {
                rng: R,
                #(#const_world_fields)*
                #(#var_fields)*
                #(#bound_log_cum_fields)*
                #(#obs_world_fields)*
            }

            impl<R: rand::Rng> Iterator for World<R> {
                type Item = Sample;

                fn next(&mut self) -> Option<Self::Item> {
                    Some(self.sample())
                }
            }

            impl<R: rand::Rng> World<R> {
                pub fn new(rng: R, #(#const_new_params)* #(#obs_new_params)*) -> World<R> {
                    World {
                        rng: rng,
                        #(#const_new_inits)*
                        #(#var_inits)*
                        #(#bound_log_cum_inits)*
                        #(#obs_new_inits)*
                    }
                }

                pub fn reset(&mut self) {
                    #(#reset_stmts)*
                    #(#bound_log_cum_reset_stmts)*
                }

                /// Like `reset`, but pins every stochastic observed variable to its
                /// known value and resets everything else (including deterministic
                /// variables) to Unknown so they are re-evaluated from the freshly
                /// sampled stochastic latents.
                ///
                /// Used by importance sampling.
                fn reset_for_weighted(&mut self) {
                    #(#reset_for_weighted_stmts)*
                    #(#bound_log_cum_reset_stmts)*
                    #(#weighted_pin_stmts)*
                }

                /// Draw one exact posterior sample via rejection sampling.
                ///
                /// Loops until a prior draw matches every observed value, then
                /// returns the queried variable values.  Only valid for
                /// discrete observations.
                pub fn sample(&mut self) -> Sample {
                    loop {
                        self.reset();
                        #(#rejection_checks)*
                        return Sample {
                            #(#query_sample_fields)*
                        };
                    }
                }

                #weighted_sample_method

                #active_eval_method

                #(#eval_methods)*
            }
        }
    }
}

fn value_type(base: &Type, dims: usize) -> TokenStream {
    let mut ty = quote! { #base };
    for _ in 0..dims {
        ty = quote! { Vec<#ty> };
    }
    ty
}

fn observed_type(base: &Type, dims: usize) -> TokenStream {
    if dims == 0 {
        quote! { #base }
    } else {
        let mut ty = quote! { Option<#base> };
        for _ in 0..dims {
            ty = quote! { Vec<#ty> };
        }
        ty
    }
}

fn cache_type(base: &Type, dims: usize) -> TokenStream {
    let mut ty = quote! { FeOption<#base> };
    for _ in 0..dims {
        ty = quote! { Vec<#ty> };
    }
    ty
}

fn const_field_ident(konst: &ConstantIR) -> Ident {
    format_ident!("const_{}", konst.const_ident)
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn index_ident(variable: &VariableIR, level: usize) -> Ident {
    variable.indices[level].index_ident.clone()
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn index_idents(variable: &VariableIR) -> Vec<Ident> {
    (0..variable.indices.len())
        .map(|level| index_ident(variable, level))
        .collect()
}

fn loop_ident(level: usize) -> Ident {
    format_ident!("__ferric_idx_{}", level)
}

fn obs_item_ident(level: usize) -> Ident {
    format_ident!("__ferric_obs_{}", level)
}

fn bound_log_cum_field_ident(variable: &VariableIR) -> Ident {
    format_ident!("bound_log_cum_{}", variable.var_ident)
}

fn bound_log_cum_eval_ident(variable: &VariableIR) -> Ident {
    format_ident!("eval_bound_log_cum_{}", variable.var_ident)
}

fn eval_at_ident(variable: &VariableIR) -> Ident {
    format_ident!("eval_at_{}", variable.var_ident)
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn bound_expr(
    upper: &Ident,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    let name = upper.to_string();
    if let Some(konst) = consts.get(&name) {
        let field = const_field_ident(konst);
        quote! { self.#field }
    } else if variables.contains_key(&name) {
        let eval = format_ident!("eval_{}", upper);
        quote! { self.#eval() }
    } else {
        quote! { #upper }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn bound_log_cum_method_for(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    if !variable.is_stochastic || variable.max_expr.is_none() {
        return quote! {};
    }

    let eval_ident = bound_log_cum_eval_ident(variable);
    let field_ident = bound_log_cum_field_ident(variable);
    let value_ty = value_type(&parse_quote!(f64), variable.indices.len());
    let eval_dist_ident = format_ident!("evaldist_{}", variable.var_ident);
    let max_expr = variable
        .max_expr
        .as_ref()
        .expect("checked max expression exists");
    let scoped_indices = index_idents(variable);
    let max_expr = replace(quote! { #max_expr }, variables, consts, &scoped_indices);

    if variable.indices.is_empty() {
        quote! {
            pub fn #eval_ident(&mut self) -> #value_ty {
                if self.#field_ident.is_unknown() {
                    let dist = self.#eval_dist_ident();
                    let max_value = #max_expr;
                    let log_cum_prob = dist.log_cum_prob(&max_value);
                    self.#field_ident = FeOption::Known(log_cum_prob);
                }
                self.#field_ident.unwrap_clone()
            }
        }
    } else {
        let idx_args = index_idents(variable);
        let body = quote! {
            let dist = self.#eval_dist_ident(#(#idx_args),*);
            let max_value = #max_expr;
            dist.log_cum_prob(&max_value)
        };
        let normalizer_array = nested_array_expr(0, variable, variables, consts, body);
        quote! {
            pub fn #eval_ident(&mut self) -> #value_ty {
                if self.#field_ident.is_unknown() {
                    let log_cum_probs = #normalizer_array;
                    self.#field_ident = FeOption::Known(log_cum_probs);
                }
                self.#field_ident.unwrap_clone()
            }
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn eval_methods_for(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    let eval_ident = format_ident!("eval_{}", variable.var_ident);
    let var_ident = format_ident!("var_{}", variable.var_ident);
    let value_ty = value_type(&variable.type_ident, variable.indices.len());
    let base_ty = &variable.type_ident;
    let dep = &variable.dependency;
    let scoped_indices = index_idents(variable);
    let dep_expr = replace(quote! { #dep }, variables, consts, &scoped_indices);

    if variable.is_stochastic {
        let eval_dist_ident = format_ident!("evaldist_{}", variable.var_ident);
        let index_params: Vec<_> = (0..variable.indices.len())
            .map(|level| {
                let ident = index_ident(variable, level);
                quote! { #ident: u64 }
            })
            .collect();
        let index_param_idents = index_idents(variable);
        let dist_method = quote! {
            pub fn #eval_dist_ident(&mut self, #(#index_params),*) -> Box<dyn ferric::distributions::Distribution<R, Domain=#base_ty>> {
                #(let _ = #index_param_idents;)*
                let dist = #dep_expr;
                Box::new(dist.unwrap())
            }
        };

        if variable.indices.is_empty() {
            let sample_expr = if let Some(max_expr) = &variable.max_expr {
                let max_expr = replace(quote! { #max_expr }, variables, consts, &[]);
                quote! {
                    let max_value = #max_expr;
                    loop {
                        let sampled = dist.sample(&mut self.rng);
                        if sampled <= max_value {
                            break sampled;
                        }
                    }
                }
            } else {
                quote! {
                    dist.sample(&mut self.rng)
                }
            };
            quote! {
                pub fn #eval_ident(&mut self) -> #value_ty {
                    if self.#var_ident.is_null() {
                        panic!(
                            "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                            stringify!(#eval_ident),
                            self.__ferric_active_evaluations()
                        );
                    }
                    if self.#var_ident.is_unknown() {
                        self.#var_ident = FeOption::Null;
                        let dist = self.#eval_dist_ident();
                        let sampled = { #sample_expr };
                        self.#var_ident = FeOption::Known(sampled);
                    }
                    self.#var_ident.unwrap_clone()
                }

                #dist_method
            }
        } else {
            let eval_at = eval_at_method_for_indexed_stochastic(variable, variables, consts);
            let cache_initializer = cache_initializer_for_variable(variable, variables, consts);
            let body = {
                let eval_at_ident = eval_at_ident(variable);
                let idx_args = index_idents(variable);
                quote! { self.#eval_at_ident(#(#idx_args),*) }
            };
            let sample_array = nested_array_expr(0, variable, variables, consts, body);
            quote! {
                pub fn #eval_ident(&mut self) -> #value_ty {
                    if self.#var_ident.is_unknown() {
                        let cache = #cache_initializer;
                        self.#var_ident = FeOption::Known(cache);
                    }
                    #sample_array
                }

                #dist_method

                #eval_at
            }
        }
    } else if variable.indices.is_empty() {
        quote! {
            pub fn #eval_ident(&mut self) -> #value_ty {
                if self.#var_ident.is_null() {
                    panic!(
                        "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                        stringify!(#eval_ident),
                        self.__ferric_active_evaluations()
                    );
                }
                if self.#var_ident.is_unknown() {
                    self.#var_ident = FeOption::Null;
                    let val = #dep_expr;
                    self.#var_ident = FeOption::Known(val);
                }
                self.#var_ident.unwrap_clone()
            }
        }
    } else {
        let eval_at = eval_at_method_for_indexed_deterministic(variable, variables, consts);
        let cache_initializer = cache_initializer_for_variable(variable, variables, consts);
        let eval_at_ident = eval_at_ident(variable);
        let idx_args = index_idents(variable);
        let body = quote! { self.#eval_at_ident(#(#idx_args),*) };
        let sample_array = nested_array_expr(0, variable, variables, consts, body);
        quote! {
            pub fn #eval_ident(&mut self) -> #value_ty {
                if self.#var_ident.is_unknown() {
                    let cache = #cache_initializer;
                    self.#var_ident = FeOption::Known(cache);
                }
                #sample_array
            }

            #eval_at
        }
    }
}

fn nested_array_expr(
    level: usize,
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
    body: TokenStream,
) -> TokenStream {
    if level == variable.indices.len() {
        return quote! { { #body } };
    }
    let loop_idx = loop_ident(level);
    let idx = index_ident(variable, level);
    let bound = bound_expr(&variable.indices[level].upper_ident, variables, consts);
    let inner = nested_array_expr(level + 1, variable, variables, consts, body);
    quote! {
        (0..(#bound as usize))
            .map(|#loop_idx| {
                let #idx = #loop_idx as u64;
                let _ = #idx;
                #inner
            })
            .collect::<Vec<_>>()
    }
}

fn cache_initializer_for_variable(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    nested_cache_expr(0, variable, variables, consts)
}

fn nested_cache_expr(
    level: usize,
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    if level == variable.indices.len() {
        return quote! { FeOption::Unknown };
    }
    let bound = bound_expr(&variable.indices[level].upper_ident, variables, consts);
    let inner = nested_cache_expr(level + 1, variable, variables, consts);
    quote! {
        (0..(#bound as usize))
            .map(|_| #inner)
            .collect::<Vec<_>>()
    }
}

fn eval_at_method_for_indexed_stochastic(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    let eval_at_ident = eval_at_ident(variable);
    let var_ident = format_ident!("var_{}", variable.var_ident);
    let eval_dist_ident = format_ident!("evaldist_{}", variable.var_ident);
    let base_ty = &variable.type_ident;
    let index_params: Vec<_> = index_idents(variable)
        .into_iter()
        .map(|ident| quote! { #ident: u64 })
        .collect();
    let idx_args = index_idents(variable);
    let access = indexed_access(quote! { cache }, variable);
    let active_label = active_label_for(variable);
    let cache_initializer = cache_initializer_for_variable(variable, variables, consts);
    let sample_from_dist = if let Some(max_expr) = &variable.max_expr {
        let max_expr = replace(quote! { #max_expr }, variables, consts, &idx_args);
        quote! {
            let max_value = #max_expr;
            loop {
                let sampled = dist.sample(&mut self.rng);
                if sampled <= max_value {
                    break sampled;
                }
            }
        }
    } else {
        quote! { dist.sample(&mut self.rng) }
    };
    quote! {
        pub fn #eval_at_ident(&mut self, #(#index_params),*) -> #base_ty {
            if self.#var_ident.is_unknown() {
                let cache = #cache_initializer;
                self.#var_ident = FeOption::Known(cache);
            }
            {
                let cache = match &self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => {
                        panic!(
                            "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                            #active_label,
                            self.__ferric_active_evaluations()
                        );
                    }
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                if #access.is_null() {
                    panic!(
                        "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                        #active_label,
                        self.__ferric_active_evaluations()
                    );
                }
                if #access.is_known() {
                    return #access.unwrap_clone();
                }
            }
            {
                let cache = match &mut self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => unreachable!("indexed variables use per-cell breadcrumbs"),
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                #access = FeOption::Null;
            }

            let dist = self.#eval_dist_ident(#(#idx_args),*);
            let sampled = { #sample_from_dist };

            {
                let cache = match &mut self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => unreachable!("indexed variables use per-cell breadcrumbs"),
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                #access = FeOption::Known(sampled.clone());
            }
            sampled
        }
    }
}

fn eval_at_method_for_indexed_deterministic(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    let eval_at_ident = eval_at_ident(variable);
    let var_ident = format_ident!("var_{}", variable.var_ident);
    let base_ty = &variable.type_ident;
    let dep = &variable.dependency;
    let scoped_indices = index_idents(variable);
    let dep_expr = replace(quote! { #dep }, variables, consts, &scoped_indices);
    let index_params: Vec<_> = scoped_indices
        .iter()
        .map(|ident| quote! { #ident: u64 })
        .collect();
    let access = indexed_access(quote! { cache }, variable);
    let active_label = active_label_for(variable);
    let cache_initializer = cache_initializer_for_variable(variable, variables, consts);
    quote! {
        pub fn #eval_at_ident(&mut self, #(#index_params),*) -> #base_ty {
            if self.#var_ident.is_unknown() {
                let cache = #cache_initializer;
                self.#var_ident = FeOption::Known(cache);
            }
            {
                let cache = match &self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => {
                        panic!(
                            "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                            #active_label,
                            self.__ferric_active_evaluations()
                        );
                    }
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                if #access.is_null() {
                    panic!(
                        "Ferric dependency loop while evaluating `{}`. Active evaluations: {:?}",
                        #active_label,
                        self.__ferric_active_evaluations()
                    );
                }
                if #access.is_known() {
                    return #access.unwrap_clone();
                }
            }
            {
                let cache = match &mut self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => unreachable!("indexed variables use per-cell breadcrumbs"),
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                #access = FeOption::Null;
            }

            let val = #dep_expr;

            {
                let cache = match &mut self.#var_ident {
                    FeOption::Known(cache) => cache,
                    FeOption::Null => unreachable!("indexed variables use per-cell breadcrumbs"),
                    FeOption::Unknown => unreachable!("indexed cache was initialized above"),
                };
                #access = FeOption::Known(val.clone());
            }
            val
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn array_weighted_log_term(
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
) -> TokenStream {
    let obs_field = format_ident!("obs_{}", variable.var_ident);
    let obs_local = format_ident!("__ferric_obs_{}", variable.var_ident);
    let bound_log_cum_local = format_ident!("__ferric_bound_log_cum_{}", variable.var_ident);
    let bound_log_cum_prelude = if variable.max_expr.is_some() {
        let eval_bound = bound_log_cum_eval_ident(variable);
        quote! { let #bound_log_cum_local = self.#eval_bound(); }
    } else {
        quote! {}
    };
    let bound_log_cum_ref = variable
        .max_expr
        .as_ref()
        .map(|_| quote! { #bound_log_cum_local });
    let start_ref = quote! { &#obs_local };
    let body =
        array_weighted_log_loop(0, variable, variables, consts, start_ref, bound_log_cum_ref);
    quote! {
        {
            let #obs_local = self.#obs_field.clone();
            #bound_log_cum_prelude
            #body
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn array_weighted_log_loop(
    level: usize,
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
    obs_ref: TokenStream,
    bound_log_cum_ref: Option<TokenStream>,
) -> TokenStream {
    if level == variable.indices.len() {
        let eval_dist = format_ident!("evaldist_{}", variable.var_ident);
        let idx_args = index_idents(variable);
        let bound_log_cum = bound_log_cum_ref
            .map(|log_cum| indexed_access(log_cum, variable))
            .or_else(|| Some(quote! { 0.0 }));
        let log_prob_expr = weighted_log_prob_expr(
            quote! { dist },
            quote! { __ferric_observed_value },
            variable,
            variables,
            consts,
            bound_log_cum,
        );
        return quote! {
            if let Some(__ferric_observed_value) = (#obs_ref).as_ref() {
                let dist = self.#eval_dist(#(#idx_args),*);
                log_weight += #log_prob_expr;
            }
        };
    }

    let zero_idx = loop_ident(level);
    let idx = index_ident(variable, level);
    let item = obs_item_ident(level);
    let bound = bound_expr(&variable.indices[level].upper_ident, variables, consts);
    let inner = array_weighted_log_loop(
        level + 1,
        variable,
        variables,
        consts,
        quote! { #item },
        bound_log_cum_ref,
    );
    quote! {
        for (#zero_idx, #item) in (#obs_ref).iter().enumerate() {
            let #idx = #zero_idx as u64;
            let _ = #idx;
            if #idx >= #bound {
                break;
            }
            #inner
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn indexed_access(base: TokenStream, variable: &VariableIR) -> TokenStream {
    let mut expr = base;
    for idx in index_idents(variable) {
        expr = quote! { #expr[#idx as usize] };
    }
    expr
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn active_label_for(variable: &VariableIR) -> String {
    if variable.indices.is_empty() {
        variable.var_ident.to_string()
    } else {
        let indices = index_idents(variable)
            .into_iter()
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}[{}]", variable.var_ident, indices)
    }
}

fn active_eval_method_for(variables: &HashMap<String, VariableIR>) -> TokenStream {
    let mut checks = Vec::<TokenStream>::new();
    for variable in variables.values() {
        let var_ident = format_ident!("var_{}", variable.var_ident);
        let label = variable.var_ident.to_string();
        if variable.indices.is_empty() {
            checks.push(quote! {
                if self.#var_ident.is_null() {
                    active.push(#label.to_string());
                }
            });
        } else {
            let collect = active_eval_collect_expr(0, variable, quote! { cache }, Vec::new());
            checks.push(quote! {
                if let FeOption::Known(cache) = &self.#var_ident {
                    let __ferric_active_label = #label;
                    #collect
                }
            });
        }
    }
    quote! {
        fn __ferric_active_evaluations(&self) -> Vec<String> {
            let mut active = Vec::<String>::new();
            #(#checks)*
            active
        }
    }
}

fn active_eval_collect_expr(
    level: usize,
    variable: &VariableIR,
    cursor: TokenStream,
    indices: Vec<Ident>,
) -> TokenStream {
    if level == variable.indices.len() {
        let index_values = indices.iter();
        return quote! {
            if #cursor.is_null() {
                active.push(format!("{}{:?}", __ferric_active_label, vec![#(#index_values),*]));
            }
        };
    }
    let idx = format_ident!("__ferric_active_idx_{}", level);
    let item = format_ident!("__ferric_active_item_{}", level);
    let mut next_indices = indices;
    next_indices.push(idx.clone());
    let inner = active_eval_collect_expr(level + 1, variable, quote! { #item }, next_indices);
    quote! {
        for (#idx, #item) in #cursor.iter().enumerate() {
            #inner
        }
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn weighted_log_prob_expr(
    dist: TokenStream,
    observed: TokenStream,
    variable: &VariableIR,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
    bound_log_cum: Option<TokenStream>,
) -> TokenStream {
    if let Some(max_expr) = &variable.max_expr {
        let max_expr = replace(quote! { #max_expr }, variables, consts, &[]);
        let bound_log_cum = bound_log_cum.expect("bounded variables need cached log CDF");
        quote! {
            {
                let __ferric_max_value = #max_expr;
                if *#observed <= __ferric_max_value {
                    #dist.log_prob(#observed) - #bound_log_cum
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    } else {
        quote! { #dist.log_prob(#observed) }
    }
}

// replace all occurrences of `var_name` in the dependency expression with `self.eval_var_name()`
#[cfg_attr(coverage_nightly, coverage(off))]
fn replace(
    dep_tokens: TokenStream,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
    scoped_idents: &[Ident],
) -> TokenStream {
    let mut dep_it = dep_tokens.into_iter().peekable();
    let mut out = TokenStream::new();

    while let Some(tt) = dep_it.next() {
        let replaced = match tt {
            TokenTree::Ident(ref i) => {
                let i_name = &i.to_string();
                if scoped_idents.iter().any(|scoped| scoped == i) {
                    TokenStream::from(tt)
                } else if variables.contains_key(i_name) {
                    if let Some(TokenTree::Group(group)) = dep_it.peek()
                        && group.delimiter() == Delimiter::Bracket
                    {
                        let group = match dep_it.next().expect("peeked group") {
                            TokenTree::Group(group) => group,
                            _ => unreachable!("peeked group"),
                        };
                        let var_ident = Ident::new(&format!("eval_at_{}", i_name), i.span());
                        let (arg_lets, args) =
                            indexed_call_args(group.stream(), variables, consts, scoped_idents);
                        quote! {{
                            #(#arg_lets)*
                            self.#var_ident(#(#args),*)
                        }}
                    } else {
                        let var_ident = Ident::new(&format!("eval_{}", i_name), i.span());
                        quote! { self.#var_ident() }
                    }
                } else if let Some(konst) = consts.get(i_name) {
                    let const_ident = const_field_ident(konst);
                    quote! { self.#const_ident }
                } else {
                    TokenStream::from(tt)
                }
            }
            TokenTree::Group(ref g) => {
                let stream = replace(g.stream(), variables, consts, scoped_idents);
                let mut group = Group::new(g.delimiter(), stream);
                group.set_span(g.span());
                TokenStream::from(TokenTree::Group(group))
            }
            other => TokenStream::from(other),
        };
        out.extend(replaced);
    }

    out
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn indexed_call_args(
    tokens: TokenStream,
    variables: &HashMap<String, VariableIR>,
    consts: &HashMap<String, ConstantIR>,
    scoped_idents: &[Ident],
) -> (Vec<TokenStream>, Vec<Ident>) {
    let mut arg_lets = Vec::<TokenStream>::new();
    let mut args = Vec::<Ident>::new();
    let mut current = TokenStream::new();
    let mut arg_index = 0usize;
    for tt in tokens {
        match &tt {
            TokenTree::Punct(p) if p.as_char() == ',' => {
                let replaced = replace(current, variables, consts, scoped_idents);
                let arg_ident = format_ident!("__ferric_index_arg_{}", arg_index);
                arg_lets.push(quote! { let #arg_ident = (#replaced) as u64; });
                args.push(arg_ident);
                arg_index += 1;
                current = TokenStream::new();
            }
            _ => current.extend(TokenStream::from(tt)),
        }
    }
    if !current.is_empty() {
        let replaced = replace(current, variables, consts, scoped_idents);
        let arg_ident = format_ident!("__ferric_index_arg_{}", arg_index);
        arg_lets.push(quote! { let #arg_ident = (#replaced) as u64; });
        args.push(arg_ident);
    }
    (arg_lets, args)
}

#[test]
fn output_is_module_item() {
    use proc_macro2::Span;
    use syn::{ItemMod, parse_quote, parse2};
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("grass"), Span::call_site()),
        use_exprs: vec![parse_quote!(ferric::distributions::Bernoulli)],
        consts: HashMap::new(),
        variables: HashMap::from([
            (
                String::from("rain"),
                VariableIR {
                    var_ident: Ident::new(&String::from("rain"), Span::call_site()),
                    indices: Vec::new(),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(Bernoulli::new(0.2)),
                    max_expr: None,
                    max_const_ident: None,
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: false,
                    order: 0,
                },
            ),
            (
                String::from("sprinkler"),
                VariableIR {
                    var_ident: Ident::new(&String::from("sprinkler"), Span::call_site()),
                    indices: Vec::new(),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(if rain {
                        Bernoulli::new(0.01)
                    } else {
                        Bernoulli::new(0.4)
                    }),
                    max_expr: None,
                    max_const_ident: None,
                    is_stochastic: true,
                    is_queried: false,
                    is_observed: true,
                    order: 1,
                },
            ),
        ]),
    };
    let rust = codegen(ir);

    assert!(parse2::<ItemMod>(rust).is_ok());
}

#[test]
fn output_with_deterministic_var_is_module_item() {
    use proc_macro2::Span;
    use syn::{ItemMod, parse_quote, parse2};
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("det"), Span::call_site()),
        use_exprs: vec![parse_quote!(ferric::distributions::Bernoulli)],
        consts: HashMap::new(),
        variables: HashMap::from([
            (
                String::from("x"),
                VariableIR {
                    var_ident: Ident::new(&String::from("x"), Span::call_site()),
                    indices: Vec::new(),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(Bernoulli::new(0.5)),
                    max_expr: None,
                    max_const_ident: None,
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: false,
                    order: 0,
                },
            ),
            (
                String::from("two_x"),
                VariableIR {
                    var_ident: Ident::new(&String::from("two_x"), Span::call_site()),
                    indices: Vec::new(),
                    type_ident: parse_quote!(u8),
                    dependency: parse_quote!(2u8 * x as u8),
                    max_expr: None,
                    max_const_ident: None,
                    is_stochastic: false,
                    is_queried: false,
                    is_observed: true,
                    order: 1,
                },
            ),
        ]),
    };
    let rust = codegen(ir);

    // weighted_sample_iter method must NOT be generated since two_x (deterministic) is observed
    let rust_str = rust.to_string();
    assert!(
        !rust_str.contains("fn weighted_sample_iter"),
        "weighted_sample_iter should not be generated when a deterministic variable is observed"
    );
    assert!(parse2::<ItemMod>(rust).is_ok());
}

#[test]
fn output_with_indexed_deterministic_var_is_module_item() {
    use proc_macro2::Span;
    use syn::{ItemMod, parse_quote, parse2};
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("indexed_det"), Span::call_site()),
        use_exprs: vec![],
        consts: HashMap::from([(
            String::from("n"),
            ConstantIR {
                const_ident: Ident::new(&String::from("n"), Span::call_site()),
                type_ident: parse_quote!(u64),
                order: 0,
            },
        )]),
        variables: HashMap::from([(
            String::from("x"),
            VariableIR {
                var_ident: Ident::new(&String::from("x"), Span::call_site()),
                indices: vec![crate::analyze::IndexRangeIR {
                    index_ident: Ident::new(&String::from("row"), Span::call_site()),
                    upper_ident: Ident::new(&String::from("n"), Span::call_site()),
                }],
                type_ident: parse_quote!(u64),
                dependency: parse_quote!(row),
                max_expr: None,
                max_const_ident: None,
                is_stochastic: false,
                is_queried: true,
                is_observed: false,
                order: 1,
            },
        )]),
    };
    let rust = codegen(ir);

    assert!(parse2::<ItemMod>(rust).is_ok());
}

#[test]
fn output_with_indexed_stochastic_observation_and_max_is_module_item() {
    use crate::analyze::IndexRangeIR;
    use proc_macro2::Span;
    use syn::{ItemMod, parse_quote, parse2};

    let n_ident = Ident::new(&String::from("n"), Span::call_site());
    let flips_ident = Ident::new(&String::from("flips"), Span::call_site());
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("indexed_obs"), Span::call_site()),
        use_exprs: vec![
            parse_quote!(ferric::distributions::Bernoulli),
            parse_quote!(ferric::distributions::Poisson),
        ],
        consts: HashMap::from([(
            String::from("bias"),
            ConstantIR {
                const_ident: Ident::new(&String::from("bias"), Span::call_site()),
                type_ident: parse_quote!(f64),
                order: 0,
            },
        )]),
        variables: HashMap::from([
            (
                String::from("n"),
                VariableIR {
                    var_ident: n_ident.clone(),
                    indices: Vec::new(),
                    type_ident: parse_quote!(u64),
                    dependency: parse_quote!(Poisson::new(4.0)),
                    max_expr: Some(parse_quote!(5)),
                    max_const_ident: None,
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: true,
                    order: 1,
                },
            ),
            (
                String::from("flips"),
                VariableIR {
                    var_ident: flips_ident,
                    indices: vec![IndexRangeIR {
                        index_ident: Ident::new(&String::from("flip"), Span::call_site()),
                        upper_ident: n_ident,
                    }],
                    type_ident: parse_quote!(u64),
                    dependency: parse_quote!(Poisson::new(bias)),
                    max_expr: Some(parse_quote!(5)),
                    max_const_ident: None,
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: true,
                    order: 2,
                },
            ),
        ]),
    };
    let rust = codegen(ir);
    let rust_str = rust.to_string();

    assert!(rust_str.contains("log_cum_prob"));
    assert!(rust_str.contains("__ferric_bound_log_cum_flips"));
    assert!(rust_str.contains("MaskedEq"));
    assert!(parse2::<ItemMod>(rust).is_ok());
}

#[test]
fn helper_functions_cover_higher_indices_and_unknown_bounds() {
    use proc_macro2::Span;
    use syn::parse_quote;

    let unknown = Ident::new("unknown", Span::call_site());
    assert_eq!(
        bound_expr(&unknown, &HashMap::new(), &HashMap::new()).to_string(),
        "unknown"
    );

    assert_eq!(
        value_type(&parse_quote!(bool), 2).to_string(),
        "Vec < Vec < bool > >"
    );
    assert_eq!(
        observed_type(&parse_quote!(bool), 1).to_string(),
        "Vec < Option < bool > >"
    );

    let scalar = VariableIR {
        var_ident: Ident::new("scalar", Span::call_site()),
        indices: Vec::new(),
        type_ident: parse_quote!(bool),
        dependency: parse_quote!(true),
        max_expr: None,
        max_const_ident: None,
        is_stochastic: false,
        is_queried: false,
        is_observed: false,
        order: 0,
    };
    assert_eq!(active_label_for(&scalar), "scalar");
}
