// Copyright 2022 The Ferric AI Project Developers
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{Ident, Type};

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

    // All variables — used for World struct fields and reset().
    let mut var_idents = Vec::<Ident>::new(); // var_<variable name>
    let mut var_type_idents = Vec::<Type>::new(); // <variable's type>

    // Stochastic variables — generate eval_<var> + evaldist_<var>.
    let mut stoch_eval_idents = Vec::<Ident>::new(); // eval_<var>
    let mut stoch_var_idents = Vec::<Ident>::new(); // var_<var>
    let mut stoch_type_idents = Vec::<Type>::new();
    let mut stoch_eval_dist_idents = Vec::<Ident>::new(); // evaldist_<var>
    let mut stoch_eval_dist_exprs = Vec::<TokenStream>::new();

    // Deterministic variables — generate eval_<var> only (direct expression, no sampling).
    let mut det_eval_idents = Vec::<Ident>::new(); // eval_<var>
    let mut det_var_idents = Vec::<Ident>::new(); // var_<var>
    let mut det_type_idents = Vec::<Type>::new();
    let mut det_eval_exprs = Vec::<TokenStream>::new();

    // Query variables.
    let mut query_idents = Vec::<Ident>::new(); // <variable name>
    let mut query_type_idents = Vec::<Type>::new(); // <variable's type>
    let mut query_eval_var_idents = Vec::<Ident>::new(); // eval_<variable name>

    // All observed variables — Model struct fields, World::new params, rejection sampling.
    let mut obs_idents = Vec::<Ident>::new(); // <variable name>
    let mut obs_type_idents = Vec::<Type>::new(); // <variable's type>
    let mut obs_obs_idents = Vec::<Ident>::new(); // obs_<variable name>  (World field)
    let mut obs_eval_idents = Vec::<Ident>::new(); // eval_<variable name>  (rejection sampling)

    // Stochastic observed variables only — used by weighted sampling.
    let mut obs_stoch_var_idents = Vec::<Ident>::new(); // var_<var>  (reset_for_weighted pinning)
    let mut obs_stoch_obs_idents = Vec::<Ident>::new(); // obs_<var>  (pinning + log_prob)
    let mut obs_stoch_eval_dist_idents = Vec::<Ident>::new(); // evaldist_<var>  (log_prob)

    // Vars reset to Unknown in reset_for_weighted:
    //   = all non-observed vars + deterministic observed vars.
    let mut reset_for_weighted_var_idents = Vec::<Ident>::new();

    // process all the variables in the model
    for variable in ir.variables.values() {
        let var_ident = format_ident!("var_{}", &variable.var_ident);
        let eval_var = format_ident!("eval_{}", &variable.var_ident);

        var_idents.push(var_ident.clone());
        var_type_idents.push(variable.type_ident.clone());

        if variable.is_stochastic {
            let eval_dist_var = format_ident!("evaldist_{}", &variable.var_ident);
            stoch_eval_idents.push(eval_var.clone());
            stoch_var_idents.push(var_ident.clone());
            stoch_type_idents.push(variable.type_ident.clone());
            stoch_eval_dist_idents.push(eval_dist_var.clone());
            let dep = &variable.dependency;
            stoch_eval_dist_exprs.push(replace(quote! {#dep}, &ir.variables));
        } else {
            det_eval_idents.push(eval_var.clone());
            det_var_idents.push(var_ident.clone());
            det_type_idents.push(variable.type_ident.clone());
            let dep = &variable.dependency;
            det_eval_exprs.push(replace(quote! {#dep}, &ir.variables));
        }

        if variable.is_queried {
            query_idents.push(variable.var_ident.clone());
            query_type_idents.push(variable.type_ident.clone());
            query_eval_var_idents.push(eval_var.clone());
        }

        if variable.is_observed {
            obs_idents.push(variable.var_ident.clone());
            obs_type_idents.push(variable.type_ident.clone());
            obs_obs_idents.push(format_ident!("obs_{}", &variable.var_ident));
            obs_eval_idents.push(eval_var.clone());

            if variable.is_stochastic {
                obs_stoch_var_idents.push(var_ident.clone());
                obs_stoch_obs_idents.push(format_ident!("obs_{}", &variable.var_ident));
                obs_stoch_eval_dist_idents.push(format_ident!("evaldist_{}", &variable.var_ident));
            } else {
                // Deterministic observed: reset to Unknown in reset_for_weighted so it gets
                // re-evaluated from the freshly sampled stochastic variables.
                reset_for_weighted_var_idents.push(var_ident.clone());
            }
        } else {
            reset_for_weighted_var_idents.push(var_ident.clone());
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
                    #(
                        self.#obs_idents.clone(),
                    )*
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
                #(
                    {
                        let dist = self.#obs_stoch_eval_dist_idents();
                        log_weight += dist.log_prob(&self.#obs_stoch_obs_idents);
                    }
                )*
                WeightedSample {
                    log_weight,
                    sample: Sample {
                        #(
                            #query_idents: self.#query_eval_var_idents(),
                        )*
                    },
                }
            }
        }
    } else {
        quote! {}
    };

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
                #(
                    pub #query_idents: #query_type_idents,
                )*
            }

            #weighted_structs

            /// The observed data for the model.
            pub struct Model {
                #(
                    pub #obs_idents: #obs_type_idents,
                )*
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
                        #(
                            self.#obs_idents.clone(),
                        )*
                    )
                }

                #weighted_sample_iter_method
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

                /// Like `reset`, but pins every stochastic observed variable to its
                /// known value and resets everything else (including deterministic
                /// variables) to Unknown so they are re-evaluated from the freshly
                /// sampled stochastic latents.
                ///
                /// Used by importance sampling.
                fn reset_for_weighted(&mut self) {
                    #(
                        self.#reset_for_weighted_var_idents = FeOption::Unknown;
                    )*
                    #(
                        self.#obs_stoch_var_idents = FeOption::Known(self.#obs_stoch_obs_idents.clone());
                    )*
                }

                /// Draw one exact posterior sample via rejection sampling.
                ///
                /// Loops until a prior draw matches every observed value, then
                /// returns the queried variable values.  Only valid for
                /// discrete observations.
                pub fn sample(&mut self) -> Sample {
                    loop {
                        self.reset();
                        #(
                            {
                                let sampled = self.#obs_eval_idents();
                                if self.#obs_obs_idents != sampled {
                                    continue;
                                }
                            }
                        )*
                        return Sample {
                            #(
                                #query_idents: self.#query_eval_var_idents(),
                            )*
                        };
                    }
                }

                #weighted_sample_method

                // --- Stochastic variable methods ---

                #(
                pub fn #stoch_eval_idents(&mut self) -> #stoch_type_idents {
                    if self.#stoch_var_idents.is_unknown() {
                        let dist = self.#stoch_eval_dist_idents();
                        self.#stoch_var_idents = FeOption::Known(dist.sample(&mut self.rng));
                    }
                    // TODO: handle the case when the value is Null
                    self.#stoch_var_idents.unwrap_clone()
                }

                pub fn #stoch_eval_dist_idents(&mut self) -> Box<dyn ferric::distributions::Distribution<R, Domain=#stoch_type_idents>> {
                    // TODO: handle errors in constructing the distribution object
                    let dist = #stoch_eval_dist_exprs;
                    Box::new(dist.unwrap())
                }
                )*

                // --- Deterministic variable methods (no distribution, no sampling) ---

                #(
                pub fn #det_eval_idents(&mut self) -> #det_type_idents {
                    if self.#det_var_idents.is_unknown() {
                        let val = #det_eval_exprs;
                        self.#det_var_idents = FeOption::Known(val);
                    }
                    self.#det_var_idents.unwrap_clone()
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
    use syn::{ItemMod, parse_quote, parse2};
    let ir = ModelIR {
        model_ident: Ident::new(&String::from("grass"), Span::call_site()),
        use_exprs: vec![parse_quote!(ferric::distributions::Bernoulli)],
        variables: HashMap::from([
            (
                String::from("rain"),
                VariableIR {
                    var_ident: Ident::new(&String::from("rain"), Span::call_site()),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(Bernoulli::new(0.2)),
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: false,
                },
            ),
            (
                String::from("sprinkler"),
                VariableIR {
                    var_ident: Ident::new(&String::from("sprinkler"), Span::call_site()),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(if rain {
                        Bernoulli::new(0.01)
                    } else {
                        Bernoulli::new(0.4)
                    }),
                    is_stochastic: true,
                    is_queried: false,
                    is_observed: true,
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
        variables: HashMap::from([
            (
                String::from("x"),
                VariableIR {
                    var_ident: Ident::new(&String::from("x"), Span::call_site()),
                    type_ident: parse_quote!(bool),
                    dependency: parse_quote!(Bernoulli::new(0.5)),
                    is_stochastic: true,
                    is_queried: true,
                    is_observed: false,
                },
            ),
            (
                String::from("two_x"),
                VariableIR {
                    var_ident: Ident::new(&String::from("two_x"), Span::call_site()),
                    type_ident: parse_quote!(u8),
                    dependency: parse_quote!(2u8 * x as u8),
                    is_stochastic: false,
                    is_queried: false,
                    is_observed: true,
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
