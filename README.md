# PMU-B PersonalAI

## **ABOUT ME**
---

* Name: Nattawat Ruensumrit   
* Nickname: Tai  
* Expertise programming language: Python , C / C++ /C\# , Java Script , CSS, Html   
* University: King Mongkut’s University of Technology Thonburi  
* Faculty: Computer Engineering (International curriculum)

## **Summary**
* I’m Nattawat Ruensumrit (ETPCA-S0237), a computer engineering student at King Mongkut's University of Technology Thonburi
* From all the courses I've studied, I've learned that conducting research involves multiple key components, including defining the problem, gathering data, preparing the data, building models, evaluating results, creating * * visualizations, presenting findings, and outlining potential future directions.
---

## **Content**

---

| Class | Name | WorkShop | Certificate |
| :---- | ----- | ----- | ----- |
| 1 | xPore: An AI-Powered App for Bioinformaticians | [GMM](https://github.com/Nattawat1409/PMU-B-PersonalAI/blob/main/Nattawat_GMM.ipynb) | [xPore certificate](https://drive.google.com/file/d/1Y-5VTabCylzcHWf923Wqnm_oBRDJM29W/view?usp=drive_link) |
| 2 | Learning from Biosignal | [Biosignals](https://github.com/Nattawat1409/PMU-B-PersonalAI) | [Biosignal certificate](https://drive.google.com/file/d/1U5g8yaQZ34ykRnBLxsM9hZ7YZ4V3E5A4/view?usp=sharing) |
| 3 | AI for detecting code plagiarism | [CodeCloneDetection](https://github.com/Nattawat1409/PMU-B-PersonalAI/blob/main/PMU_B_CodingAI_CodeCloneDetection_Workshop_Nattawat_Ruensumrit.ipynb) | [Code plagiarism certificate](https://drive.google.com/file/d/1fPFq4ahABPO_fFqcbzP0epCkhB9MgWpI/view?usp=sharing) |
| 4 | Mental disorder detection from social media data | [Social media](https://github.com/Nattawat1409/PMU-B-PersonalAI/blob/main/Mental_disorder.ipynb) | [Mental disorder certificate](https://drive.google.com/file/d/1XuWii63aCqyDXs-DqpnnktpmTrL-X657/view?usp=sharing) |
| 5 | BiTNet: AI for diagnosing ultrasound image | [BiTNet](https://github.com/Nattawat1409/PMU-B-PersonalAI) | [BiTNet certificate](https://drive.google.com/file/d/14ALe6ISVZY5N89QTT83SPGKf8nT2WshV/view?usp=sharing) |
| 6 | AI for arresting criminals | [ObjectDetection](https://github.com/Nattawat1409/PMU-B-PersonalAI) | [AI for arresting criminals certificate](https://drive.google.com/file/d/1UCdoH4B4ptcakfH2dI8-BC5neZu6WEpl/view?usp=sharing) |



## **📝🗒️TAKING NOTE FROM 6 LESSONS ##
# **xPore: An AI-Powered App for Bioinformaticians**

xPore is a software tool leveraging Nanopore sequencing data to analyze RNA modifications like m6A, enabling researchers to uncover molecular differences across cell types.

---

## **Problem Statement**

**Objective:** Analyze RNA modifications and their impact on health and disease using Nanopore sequencing.  
**Key Focus:** Detect and quantify modifications like m6A based on changes in electrical signals.

---

## **Key Concepts**

1. **Central Dogma:** DNA → mRNA → Proteins (regulators of health/disease).  
2. **Gene Expression:** mRNA sequencing reveals gene behavior and abnormalities.  
3. **Nanopore RNA Sequencing:** Direct RNA sequencing through electrical signal analysis.  
4. **RNA Modifications:** Chemical changes (e.g., m6A) affect cellular function and diseases.

---

## **Objectives**

1. Identify modified RNA positions (e.g., m6A sites).  
2. Quantify the modification rate (fraction of modified reads).  

---

## **Tools**

- **Supervised:** EpiNano, MINES.  
- **Unsupervised:** Tombo, Nanocompore, xPore.  

---

## **Data Processing Workflow**

### **1. Data Collection**
- **FAST5:** Raw electrical signals.  
- **FASTQ:** Basecalled sequences.  
- **FASTA:** Reference sequences.  
- **BAM/SAM:** Alignment of RNA reads to references.  

### **2. Preprocessing Steps**
1. **Direct RNA Sequencing:** Collect current intensity levels.  
2. **Base Calling:** Convert signals to RNA sequences using Guppy.  
3. **Sequence Alignment:** Map sequences using Minimap2.  
4. **Signal Alignment:** Align raw signals to RNA bases using Nanopolish.

---

## **Methodology: Bayesian Multi-Sample Gaussian Mixture Modelling**

- **Gaussian Distribution:** Captures unmodified and modified RNA signals.  
- **Bayesian Inference:** Estimates parameters for classification and modification rates.  

### **Advantages**
- Accurate representation of RNA states.  
- Multi-sample analysis for robust comparisons.  
- Parallel processing for fast execution.

---

## **Validation**

- **Samples:** WT (natural m6A) vs. KO (m6A-inactivated).  
- **Metrics:**
  - **AUC-ROC:** 86%.  
  - **Accuracy:** >95% (validated against m6ACE-Seq & DRACH motifs).  
- **Outcomes:**  
  - Identified unique m6A sites.  
  - Quantified modification rates aligned with expected values.  

---

## **Applications**

- Full transcriptome-wide m6A site detection.  
- Comparative studies across tissues, cell lines, and clinical samples.  
- Potential disease-specific RNA modification discovery.

---

## **Future Directions**

### **1. Domain-Oriented**
- Train supervised models (e.g., m6anet) using xPore outputs.  
- Explore models less reliant on signal segmentation for end-to-end analysis.  
- Improve interpretability for distinguishing modifications from errors.  

### **2. Method-Oriented**
- Integrate GMM with deep learning models (e.g., CNN).  
- Test alternative distributions for long-tailed data.  

---

## **Key Takeaways**

- **Validation:** Combine ML metrics with biological insights.  
- **Applicability:** Extend to external datasets and clinical contexts.  
- **Discovery:** Enable novel findings in RNA modification research.  
- **User Accessibility:** Open-source, easy to install, lightweight, and fast.

