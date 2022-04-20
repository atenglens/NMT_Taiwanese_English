## Independent Work Project Spring 2022 at Princeton University

### Neural Machine Translation Systems for the Taiwanese-English Text-to-Text Task
**Author**: Ashley Teng

**Adviser**: Professor Danqi Chen

**Abstract**:
As a primarily spoken language, Taiwanese is an extremely low-resource language with very little research conducted in natural language processing (NLP) and no published exploration in neural machine translation (NMT). To help preserve the Taiwanese language, we build the first-ever NMT systems for the Taiwanese-English Text-to-Text task. We first create a new dataset by extracting an online Bible in Taiwanese and English to form a parallel corpus consisting of approximately 30,000 sentence pairs. Then, we fine-tune vanilla sequence-to-sequence models with Long Short-Term Memory networks as well as various transformer models on the dataset. Our results show that the best performing model is Marian pre-trained on the Romance languages with BLEU score 31.92. We observe that smaller transformers tend to perform better and increasing a model's size does not necessarily improve performance. We also observe that models pre-trained on Asian languages similar to Taiwanese do not necessarily perform better than models pre-trained on non-Asian languages, and better performance is achieved with a tokenizer that does not assume words are separated by spaces.
