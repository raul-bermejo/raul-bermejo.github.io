---
title: "MLOps, the Right Way: Part II — MLflow to the Rescue"
author: raul
date: 2024-05-28 9:00:00 +0800
categories: [portfolio]
tags: [ai/ml, thought-leadership]
pin:
---

*This blogpost was [originally published on Versent's Tech Blog](https://versent.com.au/blog/mlops-the-right-way-part-ii-mlflow-to-the-rescue/).*

## Intro

In the [previous instalment of this series]({% post_url 2024-04-11-mlops-the-right-way %}), we introduced MLOps, an approach designed to streamline the transition of AI/ML algorithms into production environments. In summary, the goal of MLOps is to leverage the potential of AI/ML solutions into tangible business value. More specifically, MLOps offers frameworks and best practices (many borrowed from DevOps) to tackle key challenges such as scalability, governance, flexible deployment options and reproducibility.

In this post, I will examine one such framework, MLflow, showcasing how it sets MLOps in motion. There is a wide range of other solutions that leverage MLOps, including [h2o.ai](https://h2o.ai/), [DataRobot](https://www.datarobot.com/), and [Weights & Biases](https://wandb.ai/). However, MLflow stands out for its robustness and accessibility.

## An Overview of MLflow

As described in their documentation, 'MLflow [is] a tool for Managing the Machine Learning Lifecycle' at scale and cloud-agnostically. It addresses MLOps challenges through:

- **Scalability:** Training, hyperparameter optimisation, and execution can be parallelised to adapt your use case to larger datasets. This is particularly advantageous if you're using a parallel processing engine such as Spark.

- **ML Governance:** It centralises model metadata and tracks lineage via a Model Registry, enabling aspects such as model re-training, versioning, documentation and security.

- **Deployment:** MLflow can adapt to any infrastructure, whether a cloud or a local environment. For cloud environments, the deployment is cloud-agnostic and compatible with most vendors, e.g., Azure, Databricks, Kubernetes, or AWS SageMaker.

- **Reproducibility:** Any team member in the organisation can reproduce model runs at any point. In AI/ML workflows, reproducibility enables collaboration across the organisation and, in critical cases where an ML model crashes, it can improve service efficiency or downtime.

MLflow fosters collaboration by providing a common framework for DevOps, Data and ML Engineers, and Management. Reproducibility is essential for collaboration and AI governance, achieved through MLflow's two core concepts: **runs** and **experiments**.

A **run** corresponds to the code, metadata, and artefacts associated with training (or re-training) an AI/ML model [1], while an **experiment** [2] is the basis of a unit of work in MLflow, representing a collection of runs aimed at solving and operationalising a specific problem.

![png](../../assets/img/mlops/mlflow-experiment.png)

<div align="center">Example of an MLflow experiment and its runs. Image by author.</div>
&nbsp;

An experiment (demo_fraud-detection in the image above) should represent a business problem (or a part of one) to be solved by an AI/ML model — ideally scoped previously in the Business Goal Framing section of the AI/ML lifecycle. For a more detailed description, see the previous part of this blog series: [Part I]({% post_url 2024-04-11-mlops-the-right-way %}).

Within an experiment, we compare different AI/ML approaches to deliver the most suitable. These different approaches represent model runs — independent executions of training and evaluation of AI/ML models. Although generally the best-performing model (or run) is the one that is selected and deployed, that might not always be desirable. For example, a model with a slightly lower performance might be a lot simpler and, therefore, more explainable, which can be critical in many situations.

## Dissecting MLflow: A Technical Overview

For engineering-inclined readers, MLflow can be viewed as a suite of APIs. Overall, each of these APIs is designed to tackle some of the different challenges that MLOps practitioners face. Although MLflow has over a dozen of these APIs (including one to develop and deploy LLM solutions), the ones that provide the core functionality of MLflow are:

- **MLflow Model Registry:** A centralised Model Hub UI for AI/ML governance featuring model lineage, versioning, tagging, and access control. MLflow Model Registry is the UI where you can create and interact with ML models.

- **MLflow Models:** These models allow you to save and load models and metadata for development, and pack them for deployment.

- **MLflow Tracking:** This API records runs and experiments, including code, data, config, and outputs through metadata. The information captured by MLflow Tracking leverages governance and reproducibility [4].

![png](../../assets/img/mlops/mlflow-apis.png)

<div align="center">MLflow APIs and how they integrate with the AI/ML Lifecycle. Source: <a href="https://www.databricks.com/blog/2019/10/17/introducing-the-mlflow-model-registry.html">Databricks</a>.</div>
&nbsp;

As you can see in the image above, these different MLflow APIs integrate with different parts of the AI/ML Algorithm Lifecycle, providing Data Scientists, ML Engineers, and DevOps Engineers with the tools to productionise AI/ML algorithms. Although you can interact with MLflow through its UI, all capabilities can be accessed programmatically through a Command Line Interface and different programming languages (Python, R, Java).

## Summary & Opportunities

In this blog post, I introduced you to MLflow, a framework for leveraging MLOps, and outline its high-level and technical components to showcase how it addresses MLOps challenges: scalability, governance, deployment flexibility, and reproducibility.

A key factor behind MLflow's popularity among AI/ML practitioners is its vibrant, active open-source community [(see here)](https://mlflow.org/blog/mlflow-year-in-review). This ensures continuous improvements and adaptability, making it a go-to choice for many. Additionally, if you (like me) are a Databricks user, then you are in luck because MLflow is natively integrated into Databricks (Databricks donated MLflow to the Linux Foundation in 2020). Regardless of your cloud or local environment, dive into MLflow today to see how it can streamline your MLOps workflow and provide value to your organisation!

## References

1. [MLflow: The Complete Guide — Run:ai](https://www.run.ai/guides/machine-learning-operations/mlflow)
2. [MLflow: A Tool for Managing the Machine Learning Lifecycle — MLflow documentation](https://mlflow.org/docs/latest/index.html)
3. [Introducing MLflow Model Registry — Databricks](https://www.databricks.com/blog/2019/10/17/introducing-the-mlflow-model-registry.html)
4. [MLflow Tracking — MLflow documentation](https://mlflow.org/docs/latest/tracking.html)
