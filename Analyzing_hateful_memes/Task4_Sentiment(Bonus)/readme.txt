d. BONUS TASK: Try to predict whether or not a meme is toxic, based on the
sentiment of the caption. Is the caption enough for this task? Share your
performance. What other improvements do you think you could make?.
===============================================================

Methodology: Sentiment Detection task. Will compare the sentiment from the meme-text and the hateful label. If it is hateful, then naive expectation is sentiment should be negative.
Data: Combining all the .jsonl files into a single csv file containing, image 12140 image paths, their meme-text and hateful label(pre-annotated by facebook:(0=not-hateful, 1=hateful))
Input: Will treat the text column from the 
Model: Using a pretrained roberta model from huggingface(siebert/sentiment-roberta-large-english)

Observation: Of the 12140 data points,4506 had hateful annotation and 7634 were not-hateful. But according to our sentiment model, only 3007 of those 4506 hateful points were classified as NEGATIVE. So, if we had only used the caption sentiment, we would have achieved a 66.7% accuracy.

Scope of Improvement: Instead of a pretrained sentiment model, we can use better models or finetune. We can even leverage LLM's like GPT4 or Claude to gauge sentiment better from text. Apart from this, we can think of including the image embeddings along with the text embeddings and train a model with fused embeddings to take into account the role of the image. Again, multimodal LLM's are always an option. On a small scale, however, I would have liked to use TRL(Transformer Reinforcement Learning) to see if it improves performance. 