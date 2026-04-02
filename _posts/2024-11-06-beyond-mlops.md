---
title: "Beyond MLOps: AI/ML Ethics & Fairness"
author: raul
date: 2024-11-06 9:00:00 +0800
categories: [blog]
tags: [ai/ml, ethics]
pin:
---

*This blogpost was [originally published on Versent's Tech Blog](https://versent.com.au/blog/beyond-mlops/).*

## Introduction

In the [last blog post]({% post_url 2024-05-28-mlops-the-right-way-part-ii %}), we explored some of the aspects that are addressed by MLOps, including scalability, governance, deployment and reproducibility. We learned about the importance of the AI/ML Lifecycle and its stages: business problem framing, data processing, model development, deployment and monitoring.

By now, MLOps has rapidly become the industry standard for developing and operationalising AI/ML systems into production. However, AI/ML systems are extremely complex systems that interact with society in a manner that is unmatched by previous software systems.

In this blog post, I outline one of the more complex aspects that MLOps does not address by default¹, mainly AI/ML ethics and fairness (though there are many more!). As we will learn, factoring these considerations into your AI/ML solution is crucial for developing responsible and ethical AI/ML systems.

## Fairness

> *"If you are arrested in the U.S. today, COMPAS or an algorithm like it will likely influence if, when and how you walk free."*
> — 'Ethics for Power Algorithms' by Abe Gong

Despite Hollywood's post-apocalyptic depiction of AI, AI/ML systems are not inherently evil. They are powerful and novel tools that can be used or misused for the benefit or detriment of our society. AI/ML systems learn from the data that they are trained on, just as infants learn behaviour from the environment they grow up in. Ideally, AI/ML systems would learn from data that represents the world *as it should be* and not of the world *as it is*. Unfortunately, more often than not, AI/ML systems learn behaviour from a biased representation of the world, which is what it is now.

For example, as mentioned in the quote above, in the US, the COMPAS model is constantly used to provide recidivism risk scores (that is, the likelihood that an ex-convict will re-offend). In other words, data about an individual (e.g. age, history of violence or age at first arrest) are input into the COMPAS model, which COMPAS uses to output the recidivism risk score. Judges across the U.S. then uses the risk score to determine court sentences (e.g. parole sentencing).

![png](../../assets/img/mlops/compas-bias.png)

<div align="center">COMPAS recidivism risk scores broken down by race. Source: <a href="https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing">ProPublica</a>.</div>
&nbsp;

As several independent researchers have shown, [COMPAS suffers from both gender and racial bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing). That is, COMPAS is more likely to give a higher recidivism risk score to an African-American male than to an individual of a different ethnicity when all other input data is equal. This is an example of what is known as **algorithmic bias**, which occurs when algorithms systematically perform worse on demographic groups that have been historically under-represented².

Another instance of algorithmic bias has repeatedly occurred in facial recognition AI systems, where AI/ML models recognise Caucasian individuals better than other ethnicities, as measured by model performance. This occurs mainly because the dataset that the AI/ML model was trained on was not equally distributed across demographic groups.

To address algorithmic bias, **Fair Machine Learning (FML)** has emerged in the last decade as the field of AI/ML that aims to define and technically implement ethics into AI/ML systems. Generally, these definitions will require an AI/ML performance metric to be the same across demographic groups. For example, risk-scoring AI/ML systems might entail training the model such that its performance metrics (accuracy, precision, false negative rate, etc.) are equal across demographic factors. In other words, ensuring the model does not perform disparately higher for certain demographic factors such as age, gender, or ethnicity.

While I will not provide an in-depth introduction to Fair Machine Learning in this blog post, I have written about FML at greater length [here]({% post_url 2022-10-17-fml-part1 %}). For the more engineering-minded readers, check out some open-sourced Python packages to leverage FML, including Google's [ml-fairness-gym](https://github.com/google/ml-fairness-gym) and [Fairlearn](https://fairlearn.org/) (funded by Microsoft).

## Opportunities

Applying the techniques of FML rigorously can be time-consuming and technically challenging, especially for organisations at the start of their AI/ML journey. Fortunately, there are many opportunities for organisations to start leveraging an ethical and fair use of AI/ML, irrespective of their size or maturity. Below are some high-level questions you and your team can ask when developing AI/ML systems:

- What are the potential sources of algorithmic bias? How was the training dataset collected?
- What subset of the population will be affected by the algorithm? Did they agree to be involved in the algorithm's output?
- How are the stakeholders involved in the product? Does that affect the algorithm's architecture?
- How will the deployed AI/ML lead to decision-making? What's the level of impact on society?

This article provides a more detailed account of how bias appears at different stages of the AI/ML Lifecycle and how these risks can be mitigated. The good news is that even smaller organisations have the potential to become champions in this space and lead the way!

## Conclusion

In this blog post, we have gone beyond the standard MLOps approach and considered one of the aspects needed for developing responsible AI/ML systems, mainly ethics and model fairness. While MLOps provides a framework to streamline AI/ML systems into production, it does not contain any dedicated stages for ethical considerations (to read more about the phases of the AI/ML Lifecycle, [see here]({% post_url 2024-04-11-mlops-the-right-way %})). Other crucial aspects of Responsible AI/ML systems include autonomy, transparency and degree of societal impact (I hope to write more about some of these in the future!).

Certainly, ethics and fairness should be critical aspects of the AI/ML Lifecycle, incorporated at its beginning (ML Problem Framing) and considered at every stage. Overall, organisations should avoid a 'Deploy first, ask (ethical) questions later' strategy when developing and deploying AI/ML systems.

---

**Footnotes**

¹ In most standards or documentation regarding MLOps, one rarely finds any mention of aspects such as ethics, autonomy or transparency.

² While I will not dive deeply into bias in this article, I have written [a more extensive account of bias in AI/ML for the interested reader here]({% post_url 2022-10-17-fml-part1 %}).

## Resources

- [What is MLOps?](https://ml-ops.org/)
- [Ethics for Power Algorithms — Abe Gong](https://medium.com/@abegong/ethics-for-powerful-algorithms-1-of-4-9b5c89b63a06)
- [Fair prediction with disparate impact: A study of bias in recidivism prediction instruments](https://www.liebertpub.com/doi/10.1089/big.2016.0047)
- [AI and Morality: What is Algorithmic Fairness?](https://www.technologyreview.com/2019/10/25/132415/what-is-algorithmic-fairness/)
- [Fairlearn](https://fairlearn.org/)
