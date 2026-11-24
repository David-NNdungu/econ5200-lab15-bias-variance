# Bias, Variance and an Unbiased Cross-Validation Score

## Objective

I examine how choosing a model using its cross-validation score can make its reported performance look better than its performance on new data.

## Methodology

- I compare polynomial degrees 1 to 12 using five-fold cross-validation on 80 simulated observations.
- I evaluate the selected model on 20,000 independent observations from the same simulation.
- I use nested cross-validation to evaluate the model-selection procedure separately from the search.
- I estimate squared bias and prediction variance across 200 simulated training samples and add the known noise variance.
- I implement a general k-fold MSE function for scikit-learn models and check it with different fold counts, models and column names.
- I compare the minimum-CV choice with the one-standard-error rule.

## Key Findings

The lowest CV score selects degree 5, with an MSE of **0.1853**. On the independent observations, its MSE is **0.2330**, giving an optimism gap of **0.0477**.

Nested cross-validation gives an MSE of **0.1910**. It separates model selection from evaluation, but the estimate still depends on the sample and folds. It evaluates the selection procedure using smaller training sets than the final model.

The noise variance is **0.2025**. An estimated CV error below this floor does not mean the model removed irreducible noise. Sampling variation and model selection can produce an unusually low estimate.

Among the six degrees in the bias-variance comparison, degree 5 has the lowest estimated expected MSE, **0.2291**. Degree 12 has much higher variance, **0.8435**, and expected MSE of **1.0478**.

My general k-fold function passes the supplied checks. For degree 5, it gives an MSE of **0.1853** with five folds and **0.1921** with ten folds.

The one-standard-error rule selects degree 4. Its independent-test MSE is **0.2376**, slightly higher than degree 5’s **0.2330**. The simpler choice does not improve prediction on this particular sample.

## Files

- `lab-ch15-diagnostic.ipynb`: executed notebook, written diagnoses, required function and optional extension.
- `src/model_selection.py`: supplied module for polynomial cross-validation and nested model selection.
- `README.md`: project description and findings.

## How to Run

Open `lab-ch15-diagnostic.ipynb` in Google Colab and select **Runtime → Run all**.

The notebook uses simulated data, so no external dataset or API key is needed. It creates the `src` folder and saves the supplied module before importing it.
