---
title: "MLOps, the Right Way: Part I — What is MLOps?"
author: raul
date: 2024-04-11 9:00:00 +0800
categories: [blog]
tags: [ai/ml, thought-leadership]
pin:
---

*This blogpost was [originally published on Versent's Tech Blog](https://versent.com.au/blog/mlops-the-right-way/).*

## Intro

In recent years, the tech industry has rapidly adopted AI-driven innovation. Across various sectors like entertainment, HR, legal, financial, and agriculture, AI and Machine Learning (AI/ML) algorithms have become pivotal drivers of business growth. Moreover, recent breakthroughs in Generative AI (GenAI) have expanded the spectrum of use cases, ranging from smarter chatbots to enhanced content and code creation tools such as Chat-GPT.

In 2023, GenAI servicing endpoints grew by 500%. However, it remains an open secret within the AI/ML community that only a fraction of these AI/ML solutions successfully transition to production systems, with [surveys estimating a mere 54% success rate](https://www.gartner.com/en/newsroom/press-releases/2022-10-03-gartner-survey-reveals-80-percent-of-companies-have-invested-in-ai-but-only-54-percent-have-deployed-an-ai-project). In essence, teams often develop a 'Proof of Concept' (PoC) to demonstrate the potential business value of an AI/ML solution, yet frequently fail to operationalise tangible business value fully. The primary reasons for these AI/ML PoCs falling short of reaching production include:

- **Scalability:** While a solution may demonstrate the desired output for a small or clean dataset (e.g. those freely available on platforms like [Kaggle](https://www.kaggle.com/)), it cannot perform well on larger 'real-life' datasets.
- **Governance:** Due to inadequate monitoring and adherence to best software practices, AI/ML solutions often get lost or abandoned, particularly as team members transition to different organisations.
- **Reproducibility:** Although an AI/ML algorithm's output may be reproducible locally, it frequently cannot be replicated across team members or in production, leading to decreased confidence in the solution.
- **Over-reliance on Vendors:** Developing AI/ML algorithms within third-party platforms that don't align with the organisation's production environment always poses integration challenges. In some cases, such integration might not be even feasible, leading to a loss of invested resources.

## MLOps to the Rescue

To address the challenges above, [MLOps](https://ml-ops.org/) has emerged as the approach to streamline the process of taking AI/ML algorithms (e.g. in the form of PoCs) to production systems [1].

MLOps encompasses different areas of a business: AI/ML, DevOps and Data Engineering. Integrating these three technical areas constitutes the main challenge in MLOps. For example, in addition to version-controlling code and data (as DevOps and DataOps approaches do), AI/ML solutions require version control of the AI/ML algorithms themselves.

![png](../../assets/img/mlops/aiml-venn_diagram.png)

<div align="center">MLOps sits at the intersection of AI/ML, DevOps and Data Engineering. Image by author.</div>
&nbsp;

In the last section, I outlined why many AI/ML solutions fail to provide business value. On the bright side, organisations that have embraced MLOps are experiencing improvements in efficiency and delivery. For example, by leveraging MLOps, Uber has been able to empower "a better customer experience, helping prevent safety incidents" while supporting "a large volume of model deployments on a daily basis."

## Leveraging MLOps: The AI/ML Algorithm Lifecycle

A crucial concept that drives MLOps into action is the **AI/ML Algorithm Lifecycle**. The AI/ML Lifecycle is a 'divide and conquer' framework that leverages MLOps by breaking the process into a series of actionable steps. These steps can then be implemented by Data, DevOps and ML Engineers. Outlining each of these actions to achieve AI/ML deployment (see diagram above):

![png](../../assets/img/mlops/aiml_lifecycle_aws.png)

<div align="center">AWS' Well-Architected AI/ML Lifecycle. Source: <a href="https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/well-architected-machine-learning.html">AWS</a>.</div>
&nbsp;

- **Business Goal:** Arguably the most important step, here we formulate the business problem and identify what metrics we must set to assess business value. Most PoCs fail because a business goal is not precisely formulated.

- **ML Problem Framing:** In this step, we translate the business language, criteria and objectives into their technical AI/ML counterparts. We have to justify why an AI/ML algorithm is the best solution for this use case and what AI/ML architecture will be optimal to provide business value (e.g. supervised vs. unsupervised learning, LLM, …).

- **Data Processing:** In the words of Amazon's CTO Werner Vogels: 'if you don't have good data, you don't have good AI.' This step entails collecting and cleaning the data so that it's ready to train the AI/ML algorithm.

- **Model Development:** In this step, we train the AI/ML algorithm and evaluate it to understand how well it will do in production — fine-tuning it as needed. For the interested reader, I've written a [more detailed description of model development steps in this blog post]({% post_url 2024-05-28-mlops-the-right-way-part-ii.md %}).

- **Deployment and Monitoring:** Finally, the model is automatically deployed and monitored in production (ideally through a DevOps pipeline), making predictions and providing business value for the organisation.

The AI/ML lifecycle is an iterative process, allowing for back-and-forth development between steps. In reality, we can break up some of these steps further. However, this picture provides a layperson's overview of how organisations can deliver business value with an AI/ML solution.

Lastly, it is crucial to point out one of the most significant challenges in deploying AI/ML solutions — doing so ethically. Although the AI/ML Lifecycle does not directly tackle ethics and bias, ongoing efforts are addressing this challenge both in industry and academia. In summary, organisations need to be accountable and proactive, driving ethical efforts from moral values and organisational culture. If you're interested in reading further (including tips for organisations to become more data ethics-driven), I've written on ['Fairness in AI/ML' at length in this previous blogpost]({% 2022-10-17-fml-part1.md %}).

## Summary & Opportunities

In this blog post, I have introduced and motivated MLOps as an approach to streamlining the business value that AI/ML can provide to organisations. The goal of MLOps is to provide frameworks and patterns to leverage the AI/ML lifecycle, thereby maximising business value. In the [next part of this series]({% post_url 2024-05-28-mlops-the-right-way-part-ii %}), I'll dive into the implementation of MLOps through frameworks and tools, primarily exploring MLflow.

## References

1. [Databricks' Big Book of MLOps](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)
2. [Uber's CI/CD for ML Online Serving and Models](https://www.uber.com/en-AU/blog/continuous-integration-deployment-ml/)
3. [AWS' Well-Architected AI/ML Lifecycle](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/well-architected-machine-learning.html)
4. [AWS re:Invent 2023: Keynote with Dr. Werner Vogels](https://www.youtube.com/watch?v=UTRBVPvzt9w)

---

*Artificial Intelligence (AI) is the general term referring to intelligent systems. Machine Learning (ML) is a branch of AI, mainly focused on intelligent software-driven systems that produce predictions (e.g. in the form of a recommendation or classification). An example of a branch of AI that is not tackled by ML is robotics. Nevertheless, AI/ML is an umbrella term that is commonly used in the tech industry. For more information, see this article.*
