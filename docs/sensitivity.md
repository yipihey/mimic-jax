# Fractional Responses in mimic-jax

The default public language is a physical percentage response, not a raw derivative with convention-dependent units.

## Parameter response

For positive observable `O` and positive parameter `theta`, mimic-jax reports `E = d ln(O) / d ln(theta) = theta / O * dO / dtheta`. If `E = -0.7`, increasing the parameter by 1% decreases the observable by approximately 0.7% near the fiducial model.

`parameter_response_matrix` accepts an observable function of the immutable parameter PyTree and selected SAGE parameter names. It returns observable values, parameter values, raw derivatives, normalized values, validity flags, names, units, normalization convention, sign, and derivative method. Multiple observables and parameters produce the standard observable-by-parameter fractional response matrix.

Logarithmic elasticity is invalid for zero or negative observables or parameters. The default policy raises `InvalidNormalizationError`; `invalid="mask"` returns `NaN` with an explicit false validity mask. For quantities without meaningful positive logarithmic scales, choose `normalization="reference_scale_sensitivity"` and supply positive `observable_scales` and `parameter_scales`. Those scales are stored in the result and archive; mimic-jax never invents them.

`validate_parameter_response` applies several symmetric multiplicative steps and evaluates `[ln O(theta(1+delta)) - ln O(theta(1-delta))] / [ln(1+delta) - ln(1-delta)]`. Comparing multiple steps exposes truncation error and float-storage noise instead of hiding them behind one finite-difference choice.

## Historical process response

Each implemented process can be perturbed during a finite epoch as `r_i -> r_i exp(epsilon_i,k)`. `process_response_tensor` returns `R_alpha,i,k = d ln(O_alpha) / d epsilon_i,k`. A value of `0.25` means that increasing that physical process by 1% during that epoch increases the final positive observable by approximately 0.25%.

The initial API deliberately uses finite epochs, not a density called sensitivity per unit time or per unit redshift. A continuous kernel changes under a coordinate transformation unless its measure is stated. Finite-epoch responses are dimensionless and keep the question stable. `uniform_ln_scale_factor_edges` constructs bins uniform in `ln(a)` and the result stores both `ln(a)` and redshift edges, ordered forward in cosmic time but labeled naturally in redshift for figures.

`finite_epoch_magnitude_weights` normalizes `abs(R)` over epochs to summarize when a process matters. It never replaces the signed response, because cancellations and sign changes are physical information.

## Response fingerprints

`response_similarity` calculates the cosine similarity between parameter response columns using their pairwise-valid observables. Values near +1 indicate nearly indistinguishable response fingerprints, values near zero indicate distinct patterns, and values near -1 indicate approximately opposite patterns. This diagnoses identifiability separately from influence: a large response says a parameter matters, while a distinct response says the chosen observables may distinguish it from other parameters.

## Example

[`examples/sage16_fractional_responses.py`](../examples/sage16_fractional_responses.py) calculates a two-observable parameter response and a finite-epoch process response for the controlled central cooling/radio-mode/quiescent slice. Its process tensor includes the coupled `agn_heating` channel. It is an API and validation example, not a Mini-Millennium scientific conclusion.

The first population application is gated on full catalog equivalence and is described in [`mimic_jax_scientific_program.md`](mimic_jax_scientific_program.md).
