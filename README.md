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

# **Learning from Biosignals**

Biosignals are crucial for medical diagnostics, providing insights into various physiological activities. For instance, EEG monitors brain function, while ECG tracks heart activity. Traditionally, interpreting these signals is performed by doctors, but high patient volumes often result in delays. Advances in AI and machine learning now allow for remote biosignal analysis, streamlining and enhancing diagnostic efficiency.

---

## **TinySleepNet: Advancing Sleep Diagnostics**

TinySleepNet, part of the *Learning from Biosignals* initiative, utilizes deep learning techniques to analyze sleep stages using EEG signals. This lightweight approach addresses common sleep disorders like insomnia and sleep apnea, enabling faster and more accurate sleep assessments.

---

## **Biosignal Analysis Overview**

### **1. Preprocessing**
- Clean raw biosignals by removing noise and artifacts to ensure the data is ready for analysis and training.

### **2. Feature Extraction**
- Identify and extract meaningful features specific to the condition being studied. This process often requires domain expertise to craft problem-relevant features.

### **3. Model Development**
- Use machine learning algorithms to train models that establish relationships between input features and desired outcomes, such as sleep stage classification.

---

## **Application-Specific Challenges**
- **Variation Across Signals:** Biosignals differ based on activity, e.g., brain waves during sleep vs. active thinking.
- **Universal Models:** Developing a generalized model for all biosignals remains complex due to the wide variability in patterns.

---

## **Deep Learning Advantages**
Deep learning models transform raw biosignals into meaningful representations with minimal preprocessing. This reduces reliance on manual feature engineering, making tasks like classification more efficient.

---

## **Sleep Stage Scoring**

### **Sleep Stages**
- **Non-REM Sleep:**
  - **N1:** Light sleep onset.
  - **N2:** Deeper sleep with K-complexes and sleep spindles.
  - **N3:** Deep restorative sleep.
- **REM Sleep:** Associated with dreams, critical for memory and creativity.
- **Awake (W):** Periods of wakefulness.

### **Key Metrics**
- **Total Sleep Time (TST):** Combined time spent in N1, N2, N3, and REM.
- **Time in Bed (TIB):** Duration between "lights off" and "lights on."
- **Sleep Efficiency (%):** \( \text{Sleep Efficiency} = \frac{\text{TST}}{\text{TIB}} \times 100\)

---

## **Challenges and Solutions**

### **Problems**
1. Manual scoring is labor-intensive and requires expert review.
2. Complex analysis involving multiple signals (EEG, EOG, EMG, etc.).
3. Existing setups are bulky and unsuitable for home use.

### **Solutions**
1. **Single-Channel EEG:** Simplifies data collection.
2. **Deep Learning Models:** Automates classification of sleep stages.
3. **Portable Devices:** Enables at-home monitoring using compact in-ear EEGs or headbands.

---

## **Datasets**

**SleepEDF (v1):**  
- **Participants:** 20 healthy adults (Age: 28.7 ± 2.9 years).  
- **Signals:** EEG, EOG, EMG, respiration data.  
- **Files:**  
  - Signal: `*.PSG.edf` (raw biosignals).  
  - Labels: `*.Hypnogram.edf` (sleep stage annotations).  

---

## **Models**

### **DeepSleepNet**
- **Purpose:** Multi-class sleep stage scoring using single-channel EEG.
- **Architecture:**
  - **Feature Learning:** Two CNN branches for high-frequency and low-frequency features.
  - **Sequence Learning:** Bi-directional RNN captures stage transition rules.
- **Input:** 30-second EEG segments.
- **Training:**
  - **Loss:** Cross-entropy.
  - **Optimizer:** Adam.
  - **Transition Rules:** Follow AASM guidelines, ensuring realistic stage transitions.

### **TinySleepNet**
- **Purpose:** Lightweight, portable model for sleep stage scoring.
- **Enhancements:**
  - Single-branch CNN simplifies feature extraction.
  - Uni-directional RNN reduces computational complexity.
  - No pre-training required.
- **Training Improvements:**
  - **Data Augmentation:** Random EEG shifts and sequence skipping.
  - **Weighted Loss:** Handles class imbalance by prioritizing rare stages.

---

## **Comparison: DeepSleepNet vs. TinySleepNet**

| **Feature**            | **DeepSleepNet**              | **TinySleepNet**               |
|-------------------------|-------------------------------|---------------------------------|
| **CNN Architecture**    | Dual-branch (Small & Large)  | Single-branch                  |
| **RNN Type**            | Bi-directional               | Uni-directional                |
| **Pre-training**        | Class-balanced oversampling  | Not required                   |
| **Data Augmentation**   | None                         | Signal and sequence augmentation |
| **Portability**         | Less portable                | Lightweight and portable        |

---

## **Evaluation Metrics**

- **Overall Metrics:**
  - **Accuracy (ACC):** Measures overall correctness.
  - **Macro F1-Score (MF1):** Averages F1-scores across all classes.
  - **Cohen’s Kappa (κ):** Assesses inter-rater agreement beyond chance.

- **Per-Class Metrics:**
  - **Precision (PR):** \( \text{PR} = \frac{\text{TP}}{\text{TP} + \text{FP}} \)  
  - **Recall (RE):** \( \text{RE} = \frac{\text{TP}}{\text{TP} + \text{FN}} \)  
  - **F1-Score:** \( \text{F1} = \frac{2 \cdot \text{PR} \cdot \text{RE}}{\text{PR} + \text{RE}} \)

- **Visualization Tools:**
  - **Hypnogram:** Visualizes sleep stages over time.
  - **Confusion Matrix:** Compares actual and predicted sleep stages.

---

## **Conclusion**

### **Deep Learning Impact**
- Ideal for tasks with abundant training data and clear patterns, such as supervised biosignal analysis.
- Limitations: Transforming raw signals into spectrograms or image representations is not always feasible for end-to-end applications.

### **Future Directions**
1. **For Doctors:**
   - Real-time monitoring of metrics like sleep, mobility, and vitals.
   - Enable early detection of diseases.
2. **For Patients:**
   - Reduce hospital visits and associated costs.
   - Improve quality of life through early diagnosis.
3. **Model Enhancements:**
   - Transfer learning to adapt models for wearable devices.
   - Focus on early-stage disease indicators for proactive care.

