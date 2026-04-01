---
title: Unleashing the Power of Data in Decision Making
author: raul
date: 2023-11-29 9:00:00 +0800
categories: [blog]
tags: [data-science, thought-leadership]
pin:
---

*This blogpost was [originally published on Versent's Tech Blog](https://versent.com.au/blog/unleashing-the-power-of-data-in-decision-making/).*

## A Case for Bayesian Reasoning

The tech industry is rapidly embracing the importance of data-driven decisions, which has proven to accelerate scalability and agility in software delivery. However, as companies wade through oceans of data, extracting the nuggets of valuable information can be a daunting task. At the beginning of their data journey, organisations lack the digital infrastructure to extract value from data. Nonetheless, even tech-savvy organisations struggle to implement a repeatable and scalable framework to leverage effective data-driven outcomes.

In this post, we will delve into Bayesian reasoning, a statistical framework that has found success across a spectrum of domains, from [scientific discovery](https://www.nature.com/articles/s41592-021-01256-7), and [spam detection algorithms](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering), to [financial forecasting](https://www.sciencedirect.com/science/article/pii/S0169207023000961). We will also explore a business case to showcase the power of Bayesian reasoning.

## How can we Make Better Decisions Systematically?

Every day, we are overwhelmed with decisions, whether they relate to identifying the most effective marketing strategy or selecting the latest technological innovations. Most often, we don't reflect on whether these decisions are the most logical ones. This is where Bayesian reasoning comes to the rescue.

Bayesian reasoning relies on two key components. Firstly, the **prior** represents the degree of confidence we have about our hypothesis before data comes into the picture. In a business context, this prior might be our degree of confidence in the success of a marketing strategy or campaign, quantified through Key Performance Indicators (KPIs).

Secondly, the **posterior** is the degree of confidence we hold in our strategy after collecting and validating the data. For example, the posterior can signify the confidence we have in this strategy once we have gathered the data from the sales and marketing teams.

## Bayesian Reasoning in Action

![png](../../assets/img/bayesian/bayes-theorem.png)

<div align="center"> Bayes' Theorem.</div>
&nbsp;


We can calculate the posterior with Bayes' Theorem. To understand each of the terms in the formula, suppose a company is launching a new campaign to increase purchases of a new product. They don't know if a strategy with 'click ads' will be more effective than customised emails sent through their newsletter, so they'd like to use Bayesian reasoning to obtain the best outcome.

Denoting **H** as our hypothesis that a customer will purchase the product based on an ad or email click, and **D** as the data regarding clicks and emails:

- **P(D\|H):** the *likelihood*, which corresponds to the probability that the purchases come from an ad versus a newsletter email, assuming they will make a purchase. From historical data of similar products, the marketing team knows this corresponds to 30% of clicking ads versus 25% for emails.

- **P(H)** represents the *prior*, probability that a customer will purchase the product based on an ad or email click. The sales team assures us this corresponds to 5% for ads versus 8% for newsletter emails.

- **P(D)** — the probability of observing the data. This term is the most technically complex but, for practical applications of Bayesian reasoning, it is data-dependent and can be disregarded.

- **P(H\|D)** is the *posterior*, the probability of our hypothesis being true given our email and ad-clicking data. Let's label this as P(H\|D)_a and P(H\|D)_e for ads and emails respectively.

When we plug in the numbers into the formula, we obtain P(H\|D)_a = 0.3 x 0.05 = 0.015 versus P(H\|D)_e = 0.25 x 0.08 = 0.02. Through Bayes' reasoning, we can be more confident that a customised email strategy is more likely to lead to a better outcome for our recently launched product.

## Opportunities & Closing Thoughts

The business case above briefly showcases how Bayesian reasoning provides us with a systematic framework for assessing and quantifying confidence in our strategies based on data. Even without performing detailed calculations, adopting a Bayesian mindset will lead you and your organisation to better outcomes. Here are a few actionable items that will help you adopt this mindset:

- **Don't think in absolutes:** From a probabilistic point of view, the confidence in our priors can never be 0 or 1. We must remain open to revising our beliefs as new data emerges or as we encounter alternative perspectives. This includes fostering a work culture where individuals from different backgrounds can express their opinions and challenge existing priors without fear of reprisal.

- **Every data, Everywhere, All at Once:** Bayesian reasoning requires us to update our strategies as new data is collected. Therefore, capturing good-quality data is crucial to adopting this framework. Collecting good-quality data is particularly critical when dealing with data that may challenge our priors, as humans tend to pay more attention to data that confirms our existing beliefs (known as confirmation bias).

In this post, we've learned how Bayesian reasoning will lead to better business outcomes by making more effective, data-driven choices. If you'd like to learn more about the power of data and AI/ML, visit [my website](https://raul-bermejo.github.io/), where I blog about AI/ML and share the projects I'm involved in.

## Resources

- [How to Lead a More Rational Life with Bayes' Theorem](https://www.scientificamerican.com/article/how-to-lead-a-more-rational-life-with-bayes-theorem/)
- [Thinking like a Bayesian — Julia Galef (Strata + Hadoop World 2016)](https://www.youtube.com/watch?v=h1wDXQQDWTk)
- [Doing business, the Bayesian way (Part 1) — GoDataDriven](https://godatadriven.com/blog/doing-business-the-bayesian-way-part-1/)
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM)
