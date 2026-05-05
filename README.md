# Search Engines (SE) - Repository Template

This repository is a template repository for the homeworks to be developed in the Search Engines course. A.Y. 2025/2026.

The homeworks are carried out by groups of students and consists in participating to one of the labs organized yearly by [CLEF](https://www.clef-initiative.eu/) (Conference and Labs of the Evaluation Forum).

*Search Engines* is a course of the

* [Master Degree in Computer Engineering](https://degrees.dei.unipd.it/master-degrees/computer-engineering/) of the  [Department of Information Engineering](https://www.dei.unipd.it/en/), [University of Padua](https://www.unipd.it/en/), Italy.
* [Master Degree in Data Science](https://datascience.math.unipd.it/) of the  [Department of Mathematics "Tullio Levi-Civita"](https://www.math.unipd.it/en/), [University of Padua](https://www.unipd.it/en/), Italy.

*Search Engines* is part of the teaching activities of the [Intelligent Interactive Information Access (IIIA) Hub](http://iiia.dei.unipd.it/).

### Group Partecipant
- Baldan Fabio (2203580)
- Donati Davide (2206352)
- Garberino Alvise (2196387)
- Padoan Giancarlo (2188345)
- Tessari Marco (2196934)

### Organisation of the repository ###

The repository is organised as follows:

* `code`: this folder contains the source code of the developed system. See dedicated section below for more details.
* `runs`: this folder contains the runs produced by the developed system.
* `results`: this folder contains the performance scores of the runs.
* `homework-1`: this folder contains the report describing the techniques applied and insights gained.
* `homework-2`: this folder contains the final paper submitted to CLEF.
* `slides`: this folder contains the slides used for presenting the conducted project.

## Virtual environments
This project had been run on Python 3.12.13, with two different virtual environments for the execution of python code. 
The one called "venv312" (the one used for translate_queries.py and evaluate_nemotronLora.py) contains:
```
accelerate==1.13.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
attrs==26.1.0
beautifulsoup4==4.14.3
blis==1.3.3
catalogue==2.0.10
cbor==1.0.0
certifi==2026.2.25
charset-normalizer==3.4.7
click==8.3.2
cloudpathlib==0.23.0
confection==1.3.3
cuda-bindings==13.2.0
cuda-pathfinder==1.5.3
cuda-toolkit==13.0.2
cymem==2.0.13
datasets==4.8.4
de_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl#sha256=fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7
dill==0.4.1
einops==0.8.2
emoji==2.15.0
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
faiss-cpu==1.13.2
filelock==3.28.0
FlagEmbedding==1.3.5
fr_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl#sha256=7d6ad14cd5078e53147bfbf70fb9d433c6a3865b695fda2657140bbc59a27e29
frozenlist==1.8.0
fsspec==2026.2.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.13.0
idna==3.11
ijson==3.5.0
inscriptis==2.7.1
ir_datasets==0.5.11
Jinja2==3.1.6
joblib==1.5.3
lxml==6.0.4
lz4==4.4.5
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
murmurhash==1.0.15
networkx==3.6.1
numpy==2.4.4
nvidia-cublas==13.1.0.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.19.0.56
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.1
pandas==3.0.2
peft==0.19.1
preshed==3.0.13
propcache==0.4.1
protobuf==7.34.1
psutil==7.2.2
pyarrow==23.0.1
pydantic==2.13.2
pydantic_core==2.46.2
Pygments==2.20.0
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.4.4
requests==2.33.1
rich==15.0.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.1
sentence-transformers==5.4.1
sentencepiece==0.2.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
smart_open==7.6.0
soupsieve==2.8.3
spacy==3.8.14
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.3
sympy==1.14.0
thinc==8.3.13
threadpoolctl==3.6.0
tiktoken==0.12.0
tokenizers==0.22.2
torch==2.11.0
tqdm==4.67.3
transformers==5.7.0
trec-car-tools==2.6
triton==3.6.0
typer==0.24.1
typing-inspection==0.4.2
typing_extensions==4.15.0
unlzw3==0.2.3
urllib3==2.6.3
warc3-wet==0.2.5
warc3-wet-clueweb09==0.2.5
wasabi==1.1.3
weasel==1.0.0
wrapt==2.1.2
xxhash==3.6.0
yarl==1.23.0
zlib-state==0.1.12
```

The one called "venv312FlagEmb" (the one used for Bi_encoder.py and for QueryExpansorBGE-large.py) contains:
```
accelerate==1.13.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
attrs==26.1.0
beautifulsoup4==4.14.3
blis==1.3.3
catalogue==2.0.10
cbor==1.0.0
certifi==2026.4.22
charset-normalizer==3.4.7
click==8.3.3
cloudpathlib==0.24.0
confection==1.3.3
cuda-bindings==13.2.0
cuda-pathfinder==1.5.4
cuda-toolkit==13.0.2
cymem==2.0.13
datasets==4.8.5
dill==0.4.1
emoji==2.15.0
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
faiss-cpu==1.13.2
filelock==3.29.0
FlagEmbedding==1.4.0
frozenlist==1.8.0
fsspec==2026.2.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.13
ijson==3.5.0
inscriptis==2.7.1
ir_datasets==0.5.11
Jinja2==3.1.6
joblib==1.5.3
lxml==6.0.4
lz4==4.4.5
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
murmurhash==1.0.15
networkx==3.6.1
numpy==2.4.4
nvidia-cublas==13.1.0.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.19.0.56
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.2
pandas==3.0.2
peft==0.19.1
preshed==3.0.13
propcache==0.4.1
protobuf==7.34.1
psutil==7.2.2
pyarrow==24.0.0
pydantic==2.13.3
pydantic_core==2.46.3
Pygments==2.20.0
python-dateutil==2.9.0.post0
PyYAML==6.0.3
regex==2026.4.4
requests==2.33.1
rich==15.0.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.1
sentence-transformers==5.4.1
sentencepiece==0.2.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
smart_open==7.6.0
soupsieve==2.8.3
spacy==3.8.14
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.3
sympy==1.14.0
thinc==8.3.13
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.11.0
tqdm==4.67.3
transformers==4.57.6
trec-car-tools==2.6
triton==3.6.0
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
unlzw3==0.2.3
urllib3==2.6.3
warc3-wet==0.2.5
warc3-wet-clueweb09==0.2.5
wasabi==1.1.3
weasel==1.0.0
wrapt==2.1.2
xxhash==3.7.0
yarl==1.23.0
zlib-state==0.1.12
```


## Execution of the code ##
**Advice**: the following code has the only scope to provide a simple but comprehensive guide to use and the right order of execution of the programs developed in this project. The paths can vary based on where the files are saved, especially for the JSON files.
```
#!/bin/bash
#TOKEN OF HUGGING FACE TO DOWNLOAD MODELS
export HF_TOKEN="hf_XXXXXXXXXXXXX"

#VENV PYTHON 3.12
source /home/{$USER$}/seupd2526-retrix/code/src/main/java/unipd/se/py/venv312/bin/activate

#VENV PYTHON 3.12 BUT WITH FLAG EMBEDDINGS
#source /home/{$USER}/seupd2526-retrix/code/src/main/java/unipd/se/py/venv312FlagEmb/bin/activate


cd /home/{$USER}/seupd2526-retrix/code/py/

python3 translate_queries.py --input final_fr_test.json --lang fr \
    --output final_fr_TRADOTTOen_test.json

python3 translate_queries.py --input final_de_test.json --lang de \
    --output final_de_TRADOTTOen_test.json

# Remember to change the path in the program to the translated queries files
python3 QueryExpansorBGE-large.py


# Remember to change the path in the program to the expanded queries files
#python3 Bi_encoder.py --precompute-corpus-hybrid

# Example to create the reranked results file for the english test set of CLEF 2026 CheckThat! task 1 
python3 evaluate_nemotronLora.py \
--model_dir    models/reranker-nemotron-1bAarsen20252026/best \
--base_model   nvidia/llama-nemotron-rerank-1b-v2 \
--topics       final_en_test.json \
--corpus       collection_data.json \
--bm25_results bi_encoder_results_bge_large_enFINAL_topk3000.json \
--output       reranked_results_nemotronFTAarsen2526_Lora_enFINALtopk2000_biEncoderTopk3000.json \
--top_k 2000 --rerank_top 100
```

### License ###

All the contents of this repository are shared using the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

![CC logo](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)

## Organization of the `code` folder

The `code` folder serves as starter for the participation at [CheckThat! 2026](https://checkthat.gitlab.io/clef2026/task1/) at [CLEF 2026](https://clef2026.clef-initiative.eu/).