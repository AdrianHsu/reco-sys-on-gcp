# Deep Dive Into Production ML workflow for Recommender System on GCP

> 🐻🎷 Scalable Collaborative Filtering End-to-end Model deployed on Google Cloud (GCS, AI Platform, Dataflow...)

<p align=center>
<a target="_blank" href="https://npmjs.org/package/life-commit" title="NPM version"><img src="https://img.shields.io/npm/v/life-commit.svg"></a><a target="_blank" href="http://nodejs.org/download/" title="Node version"><img src="https://img.shields.io/badge/node.js-%3E=_6.0-green.svg"></a><a target="_blank" href="https://opensource.org/licenses/MIT" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a><a target="_blank" href="http://makeapullrequest.com" title="PRs Welcome"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
</p>  


## System Requirement

- Python 3.6
- tensorflow 1.15.0
- tensorflow-transform 0.21.2
- pache-beam[gcp] 2.17.0

## Billable Google Cloud Services

- ML Framework: **Tensorflow**
- Data Processing Unified Model: **Apache Beam**
- Data Pipelines, DAG analytics service: **Google Dataflow**
- ML back-end: **Google AI Platform**
- Database: **Google Cloud Storage**
- Dataset: **MovieLens 100k, MovieLens 25M**

## How-to

```Shell
$ # set up google cloud CLI tools
$ ./run-cloud.sh
```

## Tree

```Shell
(ahsu) adrianhsu:~/Desktop/reco-sys-on-gcp (main)
$ tree
.
├── README.md
├── data-extractor.py
├── gitpush.sh
├── preprocess.py
├── run-cloud.sh
├── setup.py
└── trainer
    ├── __init__.py
    └── task.py

```