# Manuscript Semantic Search (MSS)

This repository provides tools and scripts to perform semantic similarity searches across a vast collection of scientific manuscripts. It leverages pre-trained language models and efficient search techniques to help you find relevant research papers based on the meaning of your query, not just keywords.

## Features

* **Comprehensive Data:** Includes a large database of scientific manuscripts from PubMed, bioRxiv, and medRxiv.
* **Semantic Search:** Employs powerful sentence embeddings to find manuscripts semantically similar to your query.
* **Efficient Chunking:** Manages large datasets efficiently using memory-mapped embedding chunks.
* **Regular Updates:** Scripts to automatically update the manuscript database with the latest publications.
* **Interactive Visualization:** Includes a Streamlit app for interactive search and visualization of results.

## Installation (using a `uv` environment)
0. **Install `uv` (if you haven't already):**
   ```bash
   pip install uv
   ``` 
   or download the latest [release](https://docs.astral.sh/uv/getting-started/installation/): 
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

1. **Create a new `uv` environment:**
   ```bash
   uv venv create mss_env
   ```
2. **Activate the environment:**
   ```bash
   uv venv activate mss_env
   ```
3. **Install the required packages:**
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

### Updating the Manuscript Database

PubMed database is massive and has over 38 million articles. The raw xml files are available for download from the [PubMed FTP site](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/). The script `pubmed_update.py` downloads the latest PubMed updates, extracts relevant metadata, and generates embeddings using the model API. The script `biorxiv_update.py` fetches the latest preprints from bioRxiv and medRxiv, processes them, and generates embeddings using the model API. The time cost of calculating embeddings is roughly 4 days * 4 GPUs. BioRxiv and medRxiv can be calculated on a single GPU in few hours.

The extraction of metadata from xml files is also not the greatest and a lot of manuscripts are missing title/abstract. As the embeddings include both it should not be a problem for the search.

Before running the search app, you need to update the manuscript database with the latest publications. First start the embedding server by running the following command:

```bash
python model_api.py
```

Then, you can update the database using the following scripts:

* **PubMed:**
   ```bash
   python pubmed_update_api.py
   ```
   This script downloads the latest PubMed updates, extracts relevant metadata, and generates embeddings using the model API.
* **bioRxiv and medRxiv:**
   ```bash
   python biorxiv_update_api.py
   ```
   This script fetches the latest preprints from bioRxiv and medRxiv, processes them, and generates embeddings using the model API.

### Running the Search App

1. **Start the model API:**
   ```bash
   python model_api.py
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run pbmss_app.py
   ```
   This will open a web interface where you can enter your search query and explore the results.

## Examples

**Search Query:** `cryo-EM structure of measles virus fusion protein antibody complex` or 
```
Filoviruses, including Ebola and Marburg viruses, cause hemorrhagic fevers with up to 90% lethality. The viral nucleocapsid is assembled by polymerization of the nucleoprotein (NP) along the viral genome, together with the viral proteins VP24 and VP35. We employed cryo-electron tomography of cells transfected with viral proteins and infected with model Ebola virus, to illuminate assembly intermediates as well as a 9Å map of the complete intracellular assembly. This structure reveals a previously unresolved, third, and outer layer of NP complexed with VP35. The intrinsically-disordered-region together with the C-terminal domain of this outer layer of NP provides the constant-width between intracellular nucleocapsid bundles and likely functions as a flexible tether to the viral matrix protein in virion. A comparison of intracellular nucleocapsid with prior in-virion nucleocapsid structures reveals the nucleocapsid further condenses vertically in-virion. The interfaces responsible for nucleocapsid assembly are highly conserved and offer targets for broadly effective antivirals.
```

**Expected Results:** A list of manuscripts related to deep brain stimulation and Parkinson's disease, ranked by their semantic similarity to the query. The app also provides interactive visualizations to explore the relationships between the found manuscripts.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. I would also appreciate new ideas for features or improvements to the existing codebase.

## License
Dawid Zyla, 2025
MIT
