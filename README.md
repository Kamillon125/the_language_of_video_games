# The Language of Video Games
A data science analysis pipeline to analyse the vocabulary, syntax and language acquisition potential of video games, fitted to the .json output of the Video Game Dialogue Corpus\n\n

The live dashboard this was used to create is available @ www.kniew.pl

## Overview

According to Stephen Krashen's Monitor Theory languages are best acquired when a relaxed learner is consuming content that a) they're interested in and b) is slightly above their current level (i + 1). I researched this potential in the medium of video games in a small pilot study, which yielded interesting results and seems to confirm that video games, as a medium, can fit this linguistic theory. There was, however, no tool for language learners to visualise a video game's language acquisition potential, so I made one using this analysis pipeline.

## Features
--Extracting dialogues\n
--CEFR vocab analysis\n
--CEFR sentence analysis\n
--Lexical density analysis\n
--Readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade, SMOG Index, Dale-Chall Score, Lexical Density)\n
--Lexical diversity analysis (TTR and MTLD)\n
--Syntactic metrics (passive voice ratio, phrasal verb density, conditional clause ratio, subordinate clause ratio, most common adj-noun collocations, most common verb-preposition collocations, etc.)\n
--(Rudimentary) slang filtering\n
--OOV vocab analysis\n
--Analysing "learned" words from every CEFR level (words appearing 12 or more times, as per Nation 2014)\n
--Saving to a bunch of .csv's, .txt's and a main results .json

## Tech Stack & Tools
--Data sourcing: https://github.com/seannyD/VideoGameDialogueCorpusPublic\n
--NLP & Linguistics: spaCy (en_core_web_md), cefrpy, Textatistic, lexical_diversity\n
--Machine Learning: Hugging Face transformers - dksysd/cefr-classifier\n

## Repo Structure
├── JSON_RESULTS/            # This folder has all the results of my analysis - things don't actually save here though\n
├── .env.example             # HuggingFace token - should be replaced with just ".env" once cloned, though.\n
├── .gitignore               # Ignored files and directories\n
├── LICENSE                  # Project license\n
├── README.md                # Project overview, tech stack, and setup instructions\n
├── advanced_metrics.py      # Calculates passives, subordinates, conditionals, collocations, slang density and phrasal verbs\n
├── analysis.py              # Orchestrator script tying the analysis steps together. This is what you run when you have a game's dialogue data (from the Video Game Dialogue Corpus) in a .json.\n
├── analyze_level.py         # Evaluates readability metrics, top 20 words, TTR, MTLD, sentence lengths, OOV and bigrams.\n
├── cefr_classify.py         # Runs the huggingface.co/dksysd/cefr-classifier for sentence classification\n
├── cefr_words.py            # Maps words to CEFR levels, calculates acquisition index and utility scores.\n
├── density.py               # Calculates lexical density (both including and excluding proper nouns)\n
├── extracting.py            # Cleans raw corpus data by stripping ACTION, NARRATIVE, and other metadata tags\n
├── lore_filter.py           # Additional text filtering utilities that ARE CURRENTLY UNUSED, but can be used if someone wants to get a game's dialogue data without its characters' names in the dialogues.\n
└── requirements.txt         # Python dependencies (spaCy, cefrpy, Textatistic, transformers, etc.)\n

## How to Run
1. Clone the repo: git clone [https://github.com/Kamillon125/the_language_of_video_games.git](https://github.com/Kamillon125/the_language_of_video_games.git)\n
2. Install dependencies: pip install -r requirements.txt\n
3. Make sure you have a game's data.json and meta.json (from the Video Game Dialogue Corpus) in the same directory as this project.\n
4. run "python analysis.py"\n
5. It might not work. For very few games I had to change some variables. But hopefully it will run :)\n

## Methodology & Limitations
A comprehensive breakdown of the entire process can be found in Methodology_and_limitations.pdf. The document also covers the limitations and areas of improvement of this project.

## Disclaimer & Fair Use
All video game titles, cover art, character names, and dialogue data referenced in this project are the property of their respective copyright holders and are sourced, analyzed, and displayed strictly for non-commercial educational and scientific research purposes, qualifying as Fair Use under applicable copyright law.

## Citations
Krashen, S. D. (1984). Principles and practice in second language acquisition. Oxford: Pergamon Press.\n
Stephanie Rennick, Seán G. Roberts, (in press) The Video Game Dialogue Corpus. Corpora 19(1). preprint\n
Stephanie Rennick, Melanie Clinton, Elena Ioannidou, Liana Oh, Charlotte Clooney, E. T., Edward Healy, Seán G. Roberts (2023) Gender bias in video game dialogue. Royal Society Open Science 10(5). https://royalsocietypublishing.org/doi/10.1098/rsos.221095\n
Eleyan, Derar & Othman, Abed & Eleyan, Amna. (2020). Enhancing Software Comments Readability Using Flesch Reading Ease Score. Information. 11. 1-25. 10.3390/info11090430.\n
Zhou, Shixiang & Jeong, Heejin & Green, Paul. (2017). How Consistent Are the Best-Known Readability Equations in Estimating the Readability of Design Standards?. IEEE Transactions on Professional Communication. PP. 1-15. 10.1109/TPC.2016.2635720. 
Dale, E., & Chall, J. S. (1948). A Formula for Predicting Readability. Educational Research Bulletin, 27(1), 11–28. http://www.jstor.org/stable/1473169 \n
Chall, J. S., & Dale, E. (1995). Readability revisited: The new Dale-Chall readability formula. Brookline Books.\n
Figueras, N. (2012). The impact of the CEFR. ELT Journal, 66(4), 477–485. https://doi.org/10.1093/elt/ccs037 \n
Laufer, Batia. (1992). How Much Lexis is Necessary for Reading Comprehension?. 10.1007/978-1-349-12396-4_12. \n
Hu, M. & Nation, P. (2000). Unknown vocabulary density and reading comprehension. Reading in a Foreign Language, 13(1), 403-430. \n
Wolf, M. (2002). The medium of the video game. University of Texas Press. \n
Shahiwala, S., & Rahul, D. R. (2025). A qualitative analysis of gamers’ experiences with vocabulary acquisition: Insights from grounded theory. System, 135, 103864. https://doi.org/10.1016/j.system.2025.103864 \n
Nation, Paul. (2014). How much input do you need to learn the most frequent 9,000 words?. Reading in a Foreign Language. 26. 1-16. 10.64152/10125/66881. \n
Mayer, R., & Sims, V. (1994). For whom is a picture worth a thousand words? Extensions of a dual-coding theory of multimedia learning. Journal of Educational Psychology, 86, 389–401. https://doi.org/10.1037/0022-0663.86.3.389 \n
Johansson, V. (2008). Lexical diversity and lexical density in speech and writing: A developmental perspective. Working Papers, 53, 61-79. \n
Riguel, E. (2014). Phrasal verbs: Usage and acquisition. Athens Journal of Philology, 1(2), 111–126. https://doi.org/10.30958/ajp.1-2-3 \n
Richards, Brian. (1987). Type/Token Ratios: what do they really tell us?. Journal of child language. 14. 201-9. 10.1017/S0305000900012885. \n
Mccarthy, Philip & Jarvis, Scott. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. Behavior research methods. 42. 381-92. 10.3758/BRM.42.2.381. \n
Saksittanupab, P. (2024). Enhancing Vocabulary Acquisition and Retention: The Role of Spaced Repetition in Language Learning. Journal of Modern Learning Development, 9(5), 205–215. retrieved from https://so06.tci-thaijo.org/index.php/jomld/article/view/273598 \n
Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. https://doi.org/10.5281/zenodo.1212303 \n
Hengel, E. (2026). Textatistic (Version 0.0.1) [Computer software]. GitHub. https://github.com/erinhengel/Textatistic \n
Frens, J. (2026). lexical-diversity (Version 0.1.1) [Computer software]. GitHub. https://github.com/jennafrens/lexical_diversity \n
Maksym, B. (2026). cefrpy [Computer software]. GitHub. https://github.com/Maximax67/cefrpy \n
dksysd. (2024). cefr-classifier [Machine learning model]. Hugging Face. https://huggingface.co/dksysd/cefr-classifier \n
Python Software Foundation. (2026). statistics — Mathematical statistics functions. Python 3.12.3 documentation. https://docs.python.org/3/library/statistics.html \n
Python Software Foundation. (2026). runpy — Locating and executing Python modules. Python 3.12.3 documentation. https://docs.python.org/3/library/runpy.html \n
Google. (accessed 2026, May 5). Active voice. Google Developers. https://developers.google.com/tech-writing/one/active-voice \n
Grammarly. (accessed 2026, May 5). Subordinate Clause: Definition, Examples, and Usage. https://www.grammarly.com/blog/grammar/subordinate-clause/ \n
Scribbr. (accessed 2026, May 5). Conditional Sentences | Definition, Examples & Forms. https://www.scribbr.com/verbs/conditional-sentences/ \n
Foreign Service Institute. (n.d.). Foreign Language Training: Language Difficulty Rankings. U.S. Department of State. \n
Internet Archive. (2026). Wayback Machine. https://archive.org/ \n
Niewiarowski, K. (2026). “Press A to Confirm”. Gry wideo jako narzędzie przyswajania języka angielskiego w oparciu o teorię Stephena Krashena. Wyniki badań pilotażowych [In press]. Kraków: Wydawnictwo AGH. \n
Niewiarowski, K. (2026). The Language of Video Games: [Source code]. GitHub. https://github.com/Kamillon125/the_language_of_video_games (this repository) \n
Niewiarowski, K. (2026). kniew [Source code]. GitHub. https://github.com/Kamillon125/kniew \n

## To run this project, create a .env file and add your Hugging Face token:\n\n\`HF_TOKEN=hf_... \`
