ClaimBuster Dataset

Project page: https://idir.uta.edu/claimbuster/


The ClaimBuster dataset consists of six files as follows:
groundtruth.csv, (22,501 sentences)
crowdsourced.csv, (1032 sentences)
all_sentences.csv, (32,072 sentences)
2xNCS.json, (8292 sentences)
2.5xNCS.json, (9674 sentences)
3xNCS.json, (11,056 sentences)
Also, all 33 presidential debate transcript files are provided as additional files in the folder “debate_transcripts”.


The files groundtruth.csv, crowdsourced.csv, and all_sentences.csv are for the paper: A Benchmark Dataset of Check-worthy Factual Claims

Both groundtruth.csv and crowdsourced.csv files contain the following attributes.
- Sentence_id: A unique numerical identifier to identify sentences in the dataset.
- Text: A sentence spoken by a debate participant. 
- Speaker: Name of the person who verbalized the Text.
- Speaker_title: Speaker’s job at the time of the debate.
- Speaker_party: Political affiliation of the Speaker.
- File_id: Debate transcript name.
- Length: Number of words in the Text.  
- Line_number: A numerical identifier to to indicate the order of the Text in the debate transcript.
- Sentiment: Sentiment score of the Text. The score ranges from -1 (most negative sentiment) to 1 (most positive sentiment). 
- Verdict: Assigned class label (1 when the sentence is CFS,0 when the sentence is UFS, and -1 when sentence is NFS).

all_sentences.csv file contains all presidential debate sentences. It has all the features shown above except for ``Verdict''. It also includes the following attribute:
- Speaker_role: It depicts the role of the Speaker in the debate as a participant.



2xNCS.json, 2.5xNCS.json, 3xNCS.json were all gathered from groundtruth.csv and crowdsourced.csv using stricter criteria with respect to the labels assigned to them. So, we expect these sets to be of higher quality and from experience we see that they produce better models. These were used in training models in the paper: Gradient-Based Adversarial Training on Transformer Networks for Detecting Check-Worthy Factual Claims. A sentence is assigned one of two classes non-check-worthy sentence (0) or check-worthy factual sentence (1). Each file has a strict ratio of sentences in the non-checkworthy class to sentences in the check-worthy class. This ratio is denoted in the file name (i.e, 2x, 2.5x, 3x). The sentences are stored as a list of dictionaries, where each dictionary has the following attributes:
  - sentence_id: A unique numerical identifier to identify sentences in the dataset.
  - label: Assigned class label (1 when the sentence is CFS, and 0 when the sentence is NCS). 
  - text: A sentence spoken by a debate participant.



Funding:
The work is partially supported by NSF grants IIS-1408928, IIP-1565699, IIS-1719054, OIA-1937143, a Knight Prototype Fund from the Knight Foundation, and subawards from Duke University as part of a grant to the Duke Tech \& Check Cooperative from the Knight Foundation and Facebook. Any opinions, findings, and conclusions or recommendations expressed in this publication are those of the authors and do not necessarily reflect the views of the funding agencies.



References:
If you use this dataset, please cite the following papers:

@inproceedings{arslan2020claimbuster,
    title={{A Benchmark Dataset of Check-worthy Factual Claims}},
    author={Arslan, Fatma and Hassan, Naeemul and Li, Chengkai and Tremayne, Mark },
    booktitle={14th International AAAI Conference on Web and Social Media},
    year={2020},
    organization={AAAI}
}

@article{meng2020gradient,
  title={Gradient-Based Adversarial Training on Transformer Networks for Detecting Check-Worthy Factual Claims},
  author={Meng, Kevin and Jimenez, Damian and Arslan, Fatma and Devasier, Jacob Daniel and Obembe, Daniel and Li, Chengkai},
  journal={arXiv preprint arXiv:2002.07725},
  year={2020}
}


