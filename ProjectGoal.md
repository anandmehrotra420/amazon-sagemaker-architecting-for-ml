NLP Amazon Reviews

Identify the topics/complaints of negative Amazon reviews in order to develop better products and anticipate customer issues.

Dataset:

Amazon reviews - Polarity: https://course.fast.ai/dataset

* Rating (1 to 5)
* Title
* Review
* Length of review

Modeling Stragegy:

Extract the negative reviews. 

First attempt: load into Amazon Comprehend and automatically extract topics.
Second attempt: Build our own topic modeler; train our data using either LDA or NTM.
	and/or
				Re-label a subset of our data in an attempt to build more accurate models to increase classification granularity. Use our own keywords to label docs/create subsets.
				
Compare first and second attempts. Proceed from there.

Can we use adjectives and nouns in conjunction?


End Goal: Determine a discrete set of common issues with products and, when given a new nevgative review, categorize it into one or more of these topics/issues.






