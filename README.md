# Probing Cultural Signals in Large Language Models through Author Profiling

This repository includes the code used to generate our paper results (and more!).
For more information about the methods and the choice we took in their implementation, we invite you to see our paper.

## Install
To reproduce the experiments, you can simply clone this repository and install the requirements in a new virtual environment as follows:

```
git clone ValentinLafargue/CulturalProbingLLM
cd CulturalProbingLLM
python3 -m venv songenv
source songenv/bin/activate (or ./songenv/Script/activate given your setup)
pip install -r requirements.txt
```

## Datasets

In our experiments we use the following datasets:
- Spotify lyrics dataset combined with MusicBrainz metadata on the artist using their API
- Deezer lyrics with Genius API combined with the Wasabi dataset for metadata on the artist.

We also use tabular datasets to investigate the newly created fairness metrics Modality Accuracy Divergence (MAD) and Recall Divergence (RD):
- Folktables : Income, Mobility, Travel Time, Employment, Public Coverage [1]
- UCI Credit dataset [2]
- COMPAS recidivism dataset [3]
- Law School Admissions dataset [4]

## Presentation & Organization 

The github is organised this way: 

- author_profiling_script: Script used to obtain author profiling result, only change to do concerns the model and the script. 
- Emotion analysis: Notebook evaluating the emotions in lyrics, and then investigate how informative they are on the LLM's predictions.
- FairnessCriteriaEvaluation: Investigate the result of the author profiling fairness metric MAD and RD on tabular datasets. 
- Identify_non_english_text: Through non-english word ratio, langdetect and langid we detect whether the lyric has non-english segment.
- merge_dataset_script: merge the deezer and spotify dataset, and remove duplicate songs through cosine similarity.
- translation_vllm_script: script used to translated the non-English lyrics.
- word_cloud_generation_script: script used to generate the word cloud presented in our paper. 
- Result_Analysis: Notebook where the author profiling results are analyzed, most figures of the paper are created in this notebook.

Some of the results were too heavy for the Github 50Mo limit, hence we share our results on the following platforms:
- [HuggingFace datasets](https://huggingface.co/datasets/ValentinLAFARGUE/AuthorProfilingResults)
- [GoogleDrive](https://drive.google.com/drive/folders/1Z0P87BRcV18Fx9HqFy5TBt83D7ig4UJc)
  
</details>

## Open source models used

In our experiments, the temperature was set to 0 most of the time (except using the Well-defined prompt), details are given in the Result_Analysis notebook.

| Model | HF ID | Revision |
|------|------|----------|
| Qwen 2.5 7B Instruct | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| DeepSeek-R1 Distill Qwen 7B | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 916b56a44061fd5cd7d6a8fb632557ed4f724f60 |
| Llama 3.1 8B Instruct | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Gemma 3 12B IT | google/gemma-3-12b-it | 96b6f1eccf38110c56df3a15bffe176da04bfd80 |
| Ministral 8B Instruct | mistralai/Ministral-8B-Instruct-2410 | 2f494a194c5b980dfb9772cb92d26cbb671fce5a |
| Mistral Small 3.2 24B | mistralai/Mistral-Small-Instruct-2409 | 4600506f6b13c7ef89e61a54263f4c9bf483de30 |
| GPT-OSS 20B | openai/gpt-oss-20b | 6cee5e81ee83917806bbde320786a8fb61efebee |

## For each model, we used in a zero-shot setting the following prompts:

We design five prompts, organized as an incremental sequence where each new prompt extends the preceding prompt by introducing an additional instruction or constraint.

- **Regular** prompt: directly asking the model to infer the sociodemographic criteria.
- **Informed** prompt: We specify the following sentence to the model: *Use lyrical content, tone, perspective, cultural references, and language patterns to decide.*
- **Informed and expressive** prompt: We further ask for keywords and explanations from the LLM, for both gender and ethnicity.
- **Well-informed and expressive** prompt: We additionally ask the model to evaluate socio-linguistic attributes such as politeness or confidence. We consider two variants of the prompt: one with the attributes evaluation first and then sociodemographic inference, the second starts with the sociodemographic inference and then evaluate the socio-linguistic attributes.
- **Corrected informed** prompt: Using rationales results from the previous prompt results, we inform the model to avoid making consistent specific errors for the ethnicity prediction. More precisely, we add to the **Informed** prompt an additional sentence clarifying that to predict ethnicity, the model should not take into account the *theme* nor the *emotions*.

## Limitations

- The dataset does not include original lyrics due to copyright restrictions.
- The experiments are conducted exclusively on song lyrics.
- We adopt the notion of ethnicity as a culturally grounded construct following sociolinguistic literature, its operationalization through regional categories remains an approximation that simplifies complex and fluid social identities.
- Lack of representation of transgender and non-binary identities in our gender ground-truth and predictions.
- When the goal is to evaluate the models’ ability to detect cultural identities, inferences based on an artist explicitly mentioning their own name in a song are not informative. This occurred in a small number of cases. Similarly but harder to remedy it, predicting the ethnicity from one specific localization-based reference is not the goal.
- We made the assumption that lyrics were written by the singer. This can be contested for two reasons: first, the existence of ghostwriters is well known; second, a considerable amount of songs were written by multiple writers and not a singular one.

## Ethical considerations

Profiling sensitive attributes such as gender and ethnicity raises important ethical risks. Inferring sociodemographic characteristics from writing can inadvertently reinforce stereotypes, essentialize cultural expression, or encourage reductive interpretations of identity. Any observed correlations between linguistic patterns and demographic labels should be interpreted cautiously and must not be viewed as deterministic, predictive, or prescriptive. Our goal is not to classify or label real individuals but to analyze the behavior of LLMs under controlled experimental conditions and to examine how cultural signals are represented in model outputs.

## References

```
[1]: Ding, F., Hardt, M., Miller, J., and Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W., editors, Advances in Neural Information Processing Systems.313, https://github.com/socialfoundations/folktables.

[2]: Quinlan, J. R. (1987). Credit Approval, UCI Machine Learning Repository, https://doi.org/10.24432/C5FS30

[3]: Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. 2016. Machine bias-there’s software used across the country to predict future criminals. and it’s biased against blacks. ProPublica.

[4]: L.F. Wightman, H. Ramsey, and Law School Admission Council. 1998. LSAC National Longitudinal Bar Passage Study. LSAC research report series. Law School Admission Council.

```

### Citation

If this was useful in your research, please consider citing our paper:

```
@misc{lafargue2026probingculturalsignalslarge,
title={Probing Cultural Signals in Large Language Models through Author Profiling},
author={Valentin Lafargue and Ariel Guerra-Adames and Emmanuelle Claeys and Elouan Vuichard and Jean-Michel Loubes},
year={2026},
eprint={2603.16749},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2603.16749},
}
```


