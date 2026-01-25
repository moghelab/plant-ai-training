A repository of publicly available AI workbooks, tutorials and training infrastructures from across the world, relevant to plant scientists

# Learning resources

## Online free courses
### General AI-related courses
* [**Cornell University - Deep Generative Models course**](https://kuleshov-group.github.io/dgm-website/) - a freely available course by Volodymyr Kuleshov. Full video lectures [here](https://www.youtube.com/@vkuleshov/playlists) including for the course on Applied Machine Learning.
* [**Cornell University - Intro to Deep Learning course**](https://www.cs.cornell.edu/courses/cs4782/2026sp/) - a freely available course by Kilian Weinberger  and Wei-Chiu Ma. It includes a complete learning syllabus, slides from each class and relevant resources.
* [**Stanford CS229: Machine Learning**](https://cs229.stanford.edu/) - A freely available course by Emily Fox, Sanmi Koyejo and Andrew Ng. Full video lectures [here](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).
* [**USDA SCINet Training Resources**](https://scinet.usda.gov/training/free-online-training#:~:text=A%20list%20of%20free%20trainings,may%20help%20you%20get%20started) - The USDA Agricultural Research Service SCINet initiative offers a compiled list of free online computational training, sorted by topic, including Python, R, statistics, Artificial Intelligence and Machine Learning, Geospatial analysis, and more. 
* [**iGotKarmayogi**](https://www.igotkarmayogi.gov.in/#/) - Government of India website for online training of government officials.

### Plant science-specific AI/ML courses
* [**CornellX - Applications of Machine Learning in Plant Science**](https://learning.edx.org/course/course-v1:CornellX+AMLPS+2T2024/home) - A freely available edX course by Adrian Powell and Gaurav Moghe. The course provides an introduction to the fundamentals of machine learning with application examples in the context of plant science.


## Hands-on tutorials and workbooks

### Plant-focused GenAI tutorials
* [**Plant DNA LLMs**](https://github.com/zhangtaolab/Plant_DNA_LLMs) - Supplement to the [paper](https://www.cell.com/molecular-plant/fulltext/S1674-2052(24)00390-3) "PDLLMs: A group of tailored DNA large language models for analyzing plant genomes". The tutorial explains how to utilize a family of DNA foundation models for identifying core plant promoters.
* [**PlantCV**](https://plantcv.org/?utm_source=chatgpt.com) - Supplement to the [paper](https://www.biorxiv.org/content/10.1101/2025.11.19.689271v1), PlantCV v4: Image analysis software for high-throughput plant phenotyping. It aims to develop open-source tools for measuring plant traits from images. It offers a set of tutorials and workshops to teach how to implement K-means Clustering, Naive Bayes and other AI algorithms to analyse plant phenotypes.
* [**Ready-Steady-Go-AI**](https://github.com/HarfoucheLab/Ready-Steady-Go-AI) - Supplement to the [paper](https://www.sciencedirect.com/science/article/pii/S2666389921001719), Ready, Steady, Go AI: A Practical Tutorial on Fundamentals of Artificial Intelligence and Its Applications in Phenomics Image Analysis. It aims to introduce the basic principles for implementing AI and explainable AI algorithms in image-based data analysis using the PlantVillage dataset to detect and classify tomato leaf diseases and spider mites as a case study.

### General AI/ML tutorials useful for plant scientists
* [**Hugging Face - Learn**](https://huggingface.co/learn) - Open-source platform with the most comprehensive ecosystem for ML development: Transformers library, Model Hub, Datasets, Spaces for deployment, and extensive documentation with practical tutorials. Relevant courses include: [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1), [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction), [Diffusion Course](https://huggingface.co/learn/diffusion-course/unit0/1), and many others.
* [**Google — Machine Learning Crash Course**](https://developers.google.com/machine-learning/crash-course) - Interactive ML fundamentals with exercises.
* [**Kaggle Learn — Micro-courses**](https://www.kaggle.com/learn) - Short, practical courses (Python, pandas, ML, feature engineering, XGBoost, etc.).
* [**fast.ai — Practical Deep Learning for Coders**](https://course.fast.ai/) - Practical Deep Learning course with notebooks.
* [**EMBL-EBI Training — Artificial Intelligence**](https://www.ebi.ac.uk/about/our-impact/ai-and-machine-learning/) — AI course materials targeted to life sciences, including plant focused.
* [**Dive Into Deep Learning**](https://d2l.ai/chapter_preface/index.html) - Deep Learning book that covers theory and practical examples.




# Datasets

## General datasets
* [**Kaggle**](https://www.kaggle.com/datasets) - Online community for data scientists to share code and datasets. Multiple users can provide Python scripts, Jupyter Notebooks and Google Collab workflows for analyzing a given dataset.
* [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/) - Contains classic biological datasets useful for teaching basic ML concepts.

## Plant-specific datasets
### Image analysis
* [**PlantVillage**](https://www.tensorflow.org/datasets/catalog/plant_village) - Supplement to the [paper](https://arxiv.org/abs/1511.08060) An open access repository of images on plant health to enable the development of mobile disease diagnostics. A large open dataset of ~54,300 images of healthy and diseased plant leaves, spanning 14 crop species and 26 diseases. Each image is labeled with species and disease (or healthy).
* [**Open Plant Phenotype Database (OPPD)**](https://datasetninja.com/open-plant-phenotyping-database#images) - Supplement to the [paper](https://www.mdpi.com/2072-4292/12/8/1246) Open Plant Phenotype Database of Common Weeds in Denmark. A public dataset of plant images for phenotyping, featuring ~7,590 RGB images covering 47 species. It includes various imaging scenarios (leaf images, whole rosettes, etc.) with annotations, useful for tasks like leaf counting and plant detection.
* [**Open Plant Image Archive**](https://ngdc.cncb.ac.cn/opia/datasets) - Supplement to the [paper](https://pubmed.ncbi.nlm.nih.gov/37930849/) OPIA: an open archive of plant images and related phenotypic traits. A comprehensive archive aggregating many plant image datasets in one place.


### Genomic and Genetic
* [**The Arabidopsis Genome Database (AGD)**](https://db.cngb.org/genomics/arabidopsis/) - Whole-genome sequences of 1,135 natural accessions of A. thaliana, plus extensive genotype and variant data.
* [**1001 Genomes Tools**](https://1001genomes.org/tools.html) - A multi-omics dataset collection focused on A. thaliana.
* [**EnsemblPlants**](https://plants.ensembl.org/index.html) - Databases providing reference genomes for dozens of plant species, along with gene annotations and comparative genomics tools.
* [**Plant Cell Atlas**](https://www.plantcellatlas.org/tools-and-repositories.html) - Community resource that contains relevant RNA-seq, single-cell RNA-seq and genetic datasets.


# Models
## Plant-specific foundation / pretrained models
* [**scPlantLLM**](https://github.com/compbioNJU/scPlantLLM/tree/main) - a transformer-based model specifically designed for the exploration of single-cell expression atlases in plants.
* [**PlantCAD**](https://www.maizegenetics.net/plantcad) - DNA foundation model trained on plant genomes. PlantCAD enables cross-species genome annotation and deleterious mutation prediction. Also includes a link to a [Google Collab workbook](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) for checking out PlantCAD.
* [**PlantBert**](https://huggingface.co/PHENOMA/PlantDeBERTa) - Based on the [paper](https://arxiv.org/html/2506.08897v1#:~:text=breakthroughs%20in%20biomedical%20and%20clinical,grounded%20entity%20normalization) PlantBert: An Open Source Language Model for Plant Science  it works as a named entity recognition (NER) and relation extraction in plant biology text, identifying genes, traits, stresses, and molecular interactions from papers.
* [**PLLaMA (Plant-LLaMA)**](https://huggingface.co/Xianjun/PLLaMa-13b-instruct) - Based on the [paper](https://arxiv.org/abs/2401.01600#:~:text=However%2C%20their%20effectiveness%20is%20limited,Moreover%2C%20we%20have%20formed%20an) PLLaMa: An Open-source Large Language Model for Plant Science it's based on Meta’s LLaMA-2 architecture, but further trained on a database of 1.5+ million plant science articles and publications. It functions as a plant science expert, providing accurate responses to questions on plant science topics.
* [**AgriBERT**]()

## General foundation models useful for plant research
* [**ESM**](https://github.com/evolutionaryscale/esm) - EvolutionaryScale Models - protein foundation models for sequence and/or structure analysis
* [**Profluent**](https://www.profluent.bio/) - Based on the [paper](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1.full) E1: Retrieval-Augmented Protein Encoder Models - Retrieval-Augmented Protein Encoder Model, provides relevant evolutionary context during training and inference of protein sequence, structure and function.
* [**AlphaFold 2**]()
* 
 
# Tools and applications using GenAI (plant-focused)

* [**Plant Connectome**](https://plant.connectome.tools/) - PlantConnectome is a powerful resource providing insights into millions of relationships involving genes, molecules, compartments, stresses, organs, and other plant entities.
* [**FuncZyme**](https://tools.moghelab.org/funczymedb) - Tools for predicting plant enzyme function, developed using LLM-based extraction of enzyme-substrate interactions.



# Compute and training platforms
## Free Computing Resources for AI Training
* [**Google Colab**]()
* [**Kaggle Notebooks**]()
* [**Hugging Face Spaces**]()

## Academic and Public HPC Infrastructures
* [**CyVerse (formerly iPlant Collaborative)**](https://learning.cyverse.org/home/what_is_cyverse/) - NSF-funded Cloud infrastructure and open source software operated at the University of Arizona. Users can launch VMs, Docker containers or access computational resources for training AI models.
* [**USDA SCINet HPC**](https://scinet.usda.gov/about/#:~:text=The%20SCINet%20initiative%20is%20an,and%20training%20in%20scientific%20computing) - USDA Agricultural Research Service’s high-performance computing initiative that offers USDA scientists and collaborators access to HPC clusters and AI hardware.
* [**ACCESS Program**](https://access-ci.org/) - NSF-funded program that offers academic researchers allocation on national supercomputers.
* [**de.NBI / ELIXIR-DE Cloud**](https://www.denbi.de/cloud) - The de.NBI network offers services, training, and cloud resources to users in life sciences and biomedicine across Germany and Europe. Read more on the [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7607484/#:~:text=The%20German%20Network%20for%20Bioinformatics,NBI%20and%20the%20German) The de.NBI / ELIXIR-DE training platform - Bioinformatics training in Germany and across Europe within ELIXIR.

## Industry and Cloud Resources
* [**Google Earth Engine & Cloud**](https://console.cloud.google.com/earth-engine/welcome?pli=1) - Offers research credits at no cost for non-comercial use to access different kind of computational resources.
* [**NVIDIA Hardware Grants and NGC**](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/) - NVIDIA sponsors academic research via GPU hardware grants and its NVIDIA GPU Cloud (NGC) which hosts pre-configured containers for deep learning. The PI must apply for the services and they must comply a list of requirements.


Please see our manuscript, Supplementary File, for more information
