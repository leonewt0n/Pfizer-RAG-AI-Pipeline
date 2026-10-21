
post in general slack channel for pizer

post introduction in welcome to extern with linkedin, 

sept. 3 meeting very important have questions ready,

Notion meeting notes:
https://externinc.notion.site/August-27th-Manager-Meeting-Pfizer-Advanced-AI-Powered-Document-Intelligence-Externship-3c91d028b9f480f79053d616133ca600



How do Multimodal AI models work? Simple explanation
https://www.youtube.com/watch?v=WkoytlA3MoQ

https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model

Gaussain Blurs and Kernel Convolution (Blur Filters)
Kernel is filter area, Gaussian uses a bell curve, normal distribution in the kernel,
![[Pasted image 20260827161309.png]]

![[Pasted image 20260827161534.png]]

https://www.youtube.com/watch?v=C_zFhWdM4ic

A large language model (LLM) at a high level and in its simplest form is a 5 step loop. Text given to it, such as in a chatgpt prompt box, is turned into tokens by cutting up the sentence into small chunks. For instance, if word tokenization is used, each chunck is one word. The chuncks are assigned a unique numerical ID. The next step is using the token ID as an address and looking up a list of numbers (vectors) that corresponds to this ID. The training process generates these vectors, or number lists, from words in its training data. A thousand numbers can be used to represent one word! Words that are similar share similar vectors. 


used in extern response (DO NOT EDIT):

explain how LLMs work to non-technical people in 3-5 words

LLMs generate responses by trying to predict the next word based on a giant list of training data that has been converted into lists of numbers, called vectors. Words that are most probable are generated next. The more data a large language model has been trained on, the better probability an accurate or sensical response will be generated. It's important to train the models with truthful and high quality content, as giving it garbage information will produce garbage outputs.

https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037

https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens


![[Pasted image 20260827153404.png]]
https://www.youtube.com/watch?v=fLvJ8VdHLA0

NLP is taking unstructured data and turning it into structured data
Stemming in tokenization is finding the root of a word, ie, the stem of running would be run.
lemmatization: using a dictionary definition  of words to better tokenize words like universe and university.
better is derived from good.
Root or lemm of better is good. 
Stemming or lemmatization for each token depending on word.
Part of speech taggin; make can be a verb or noun. The make of a laptop is a noun, vs will you make dinner?
N.E.R. : Named Entity recognition, for a given token is there an entity associated with it,
Token "Arizona" has an entity of a US state, 


SpaCy is a free, open-source library for advanced **Natural Language Processing** (NLP) in Python. It’s designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems, or to pre-process text for deep learning.

SpaCy Demo
https://demos.explosion.ai/displacy-ent?text=Jane%20Smith%20has%20secured%20a%20loan%20of%20%24300%2C000%20at%20an%20interest%20rate%20of%204.2%25%2C%20effective%20from%20March%2015%2C%202023&model=en_core_web_sm&ents=person%2Corg%2Cgpe%2Cloc%2Cproduct%2Cnorp%2Cdate%2Cper%2Cmisc

https://explosion.ai/software#spacy
















