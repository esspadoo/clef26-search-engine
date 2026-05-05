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
* `Analisi_Run.ods`: this file contains the details and some notes of the runs produced by the developed system. We chose this over other evaluation systems for the freedom and flexibility that it gives on noting and writing custom measures and notes
* `results`: this folder contains the performance scores of the runs, fine-tuned models (and the code that brought to them) and some official/custom scripts to be able to score and upload our results on CLEF's systems.
* `homework-1`: this folder contains the report describing the techniques applied and insights gained.
* `homework-2`: this folder contains the final paper submitted to CLEF.
* `slides`: this folder contains the slides used for presenting the conducted project.

## Virtual environments
This project had been run on Python 3.12.13, with two different virtual environments for the execution of python code. 

- The one called "venv312" (used for translate_queries.py and evaluate_nemotronLora.py)

- The one called "venv312FlagEmb" (used for Bi_encoder.py and for QueryExpansorBGE-large.py)

To create the desired environment go to `/code/environment/{venv312 || venv312FlagEmb}` and run the command:
`python -m venv .`

After the creation, activate it and install the required dependencies provided in the respective requirements files, running the command: `python3 -m pip install -r requirements_{VENV_NAME}.txt`

## Execution of the code ##
**Advice**: the following code has the only scope to provide a simple but comprehensive guide to use and the right order of execution of the programs developed in this project. The paths of programs' flags and absolute paths can vary based on where the files are saved and your personal configurations,especially for the JSON files. This project is meant for people who have at least a basic comprehension of search engines, information retrieval, python and Linux-based systems, not for the absolute beginner.
```
#!/bin/bash
#TOKEN OF HUGGING FACE TO DOWNLOAD MODELS
export HF_TOKEN="hf_XXXXXXXXXXXXX"

#VENV PYTHON 3.12
source seupd2526-retrix/code/environment/venv312/bin/activate

#VENV PYTHON 3.12 BUT WITH FLAG EMBEDDINGS
#source seupd2526-retrix/code/environment/venv312FlagEmb/bin/activate


cd seupd2526-retrix/code/py/

# Remember to change the path of the right files to be able to compile them with this commands
python3 translate_queries.py --input final_fr_test.json --lang fr \
    --output final_fr_TRADOTTOen_test.json


python3 translate_queries.py --input final_de_test.json --lang de \
    --output final_de_TRADOTTOen_test.json


# Remember to check the paths inside the programs to be sure that you are expanding or evaluation the right files
python3 QueryExpansorBGE-large.py


# Remember to change the path in the program to the expanded queries files
#python3 Bi_encoder.py --precompute-corpus-hybrid

# Example to create the reranked results file for the english test set of CLEF 2026 CheckThat! task 1 
python3 evaluate_nemotronLora.py \
--model_dir    results/fineTuned_models/fineTune_nemotron/nemotronFT_Train2026-All2025/ \
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