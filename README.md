<h1 align="center">Multilingual Machine Translation with T5</h1>

<p align="center">
Japanese ⇄ English ⇄ Russian Translation System  
Built with Hugging Face Transformers (T5)
</p>

<hr>

<h2>Project Overview</h2>

<p>
This project implements a multilingual neural machine translation system
using the T5 model from Hugging Face Transformers.
</p>

<p>
The system supports dynamic selection of translation directions among:
</p>

<ul>
<li>Japanese ↔ English</li>
<li>English ↔ Russian</li>
<li>Japanese ↔ Russian</li>
</ul>

<p>
The entire pipeline — from preprocessing to training, evaluation, and inference —
is implemented in a modular and reproducible manner.
</p>

<hr>

<h2>Technical Highlights</h2>

<ul>
<li>Transformer-based sequence-to-sequence architecture (T5)</li>
<li>Custom dataset batching with attention mask handling</li>
<li>Automatic train/validation/test split (9:0.5:0.5)</li>
<li>Evaluation using BLEU and RIBES</li>
<li>Reusable training and inference interface</li>
<li>Separation of concerns via modular design</li>
</ul>

<hr>

<h2>Project Structure</h2>

<pre>
HFTransformersT5Translation_jp-en-ru
├ requirements.txt
├ loaded_data
├ saved_model
├ src
│  ├ dataset.py        # Batch construction
│  ├ load.py           # Model & tokenizer loading
│  ├ preprocessing.py  # Data split (9:0.5:0.5)
│  ├ train.py          # Training pipeline
│  ├ t5_test.py           # Evaluation (BLEU, RIBES)
│  ├ translate.py      # Inference script
│  └ metrics.py        # Metric implementation
└ translation_corpus   # Hosted on Hugging Face Hub
</pre>

<hr>

<h2>Dataset</h2>

<p>
Due to file size limitations, the corpus is hosted on Hugging Face Hub:
</p>

<p>
<a href="https://huggingface.co/datasets/SHSK0118/HFTransformersT5Translation_jp-en-ru">
HFTransformersT5Translation_jp-en-ru Dataset
</a>
</p>

<p>
Each language pair contains up to 500,000 aligned sentence pairs.
</p>

<hr>

<h2>How to Run</h2>

<h3>Install dependencies</h3>

<pre>
pip install -r requirements.txt
</pre>

<h3>Train</h3>

<pre>
python src/train.py
</pre>

<h3>Evaluate</h3>

<pre>
python src/t5_test.py
</pre>

<h3>Translate interactively</h3>

<pre>
python src/translate.py
</pre>

<hr>

<h2>Evaluation Metrics</h2>

<ul>
<li><strong>BLEU</strong> – N-gram precision-based metric</li>
<li><strong>RIBES</strong> – Word order-sensitive metric</li>
</ul>

<p>
Both metrics are computed at corpus level.
</p>

<hr>

<h2>Design Philosophy</h2>

<ul>
<li>Readable and modular source structure</li>
<li>Clear separation between data processing, training, and inference</li>
<li>Reproducible experimental setup</li>
<li>Extensible to additional language pairs</li>
</ul>

<hr>

<h2>Purpose of This Project</h2>

<p>
This project was developed as a self-directed implementation to deepen
understanding of Transformer-based multilingual translation systems.
</p>

<p>
It demonstrates:
</p>

<ul>
<li>Deep learning model integration</li>
<li>End-to-end ML pipeline design</li>
<li>Metric-based evaluation</li>
<li>Multilingual NLP handling</li>
</ul>

<hr>

<h2>Author</h2>

<p>
Shota Tokunaga  
Completed Master's program | NLP / Machine Learning
</p>
