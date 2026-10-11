# Predictive virtual trials using medical imaging: a short review

*Scope:* this review covers in silico, or "virtual", clinical trials (VCTs) that use
medical images as their substrate. It looks at two families: trials of **imaging systems**
(does a new scanner or reconstruction find more disease?) and trials of **image-guided
therapy**, radiotherapy in particular (does a new treatment strategy help more patients?).
It ends with what this means for image-based dose prediction models such as DoseVLM.

---

## 1. What a predictive virtual trial is

A conventional trial gives an intervention to real patients and measures the outcome.
A virtual trial swaps each part for a computational stand-in:

| Trial component        | Virtual counterpart                                                        |
|------------------------|----------------------------------------------------------------------------|
| Patient cohort         | Computational phantoms, or digital twins built from real patient images    |
| Intervention           | Simulated device, acquisition protocol, or treatment plan                  |
| Outcome measurement    | Simulated reader / AI observer, or a dose-response / outcome model         |
| Statistical analysis   | Same endpoints and power analysis, run on many cheap simulated subjects    |

The trial is **predictive** when its endpoint is an estimate of what would happen to
patients (detection rate, toxicity, tumour control) and not just a physical quantity.
How much the endpoint can be trusted depends on the weakest model in the chain.

Possible benefits: lower cost, the ground truth is known, there are no radiation or
ethical limits on repeat "scans", rare subgroups can be enriched, and many arms can be
compared on the same virtual patients (a perfectly matched, paired design).
Recent reviews cover the field in detail (Abadi et al., 2020).

## 2. Building blocks

### 2.1 Virtual patients
* **Computational phantoms.** The XCAT family (Segars et al., 2010) and similar
  anatomical models give realistic anatomy with known organ labels and can be varied
  across body habitus, age and pathology.
* **Patient-derived models.** Segmented CT/MRI/PET of real patients. These keep the
  real anatomical variation but carry over segmentation error, and the ground truth is
  only partly known.
* **Generative models.** GANs and diffusion models can synthesise new anatomies or
  lesions. They are useful for scale, but realism and coverage of rare variants need to
  be shown, not assumed.
* **Lesion and disease models.** Inserted lesions, or mechanistic tumour growth models
  calibrated to serial imaging (Yankeelov et al., 2013), which turn a static phantom into
  a patient that changes over time.

### 2.2 Simulated intervention
* **Imaging:** Monte Carlo or analytic simulation of the acquisition physics (x-ray,
  CT, DBT, PET/SPECT, MRI), plus the vendor's reconstruction.
* **Radiotherapy:** treatment planning, often automated (knowledge-based planning,
  multicriteria optimisation) so that every virtual patient gets a consistent plan for
  each arm, and dose calculation.

### 2.3 Outcome model
* **Imaging:** model observers (ideal, channelised Hotelling) or trained AI readers
  stand in for radiologists in ROC/AUC studies.
* **Therapy:** NTCP/TCP models (e.g. Lyman–Kutcher–Burman), radiomics/dosiomics outcome
  models, or mechanistic response models that map dose and imaging features to clinical
  endpoints.

## 3. Landmark examples

* **VICTRE (FDA).** A fully in silico imaging trial with about 3,000 virtual breast
  phantoms, comparing digital breast tomosynthesis with digital mammography
  (Badano et al., 2018). It reproduced the direction and size of the DBT advantage seen
  in the comparative human trial used for regulatory approval, at a fraction of the cost
  and time. It remains the reference proof of concept that a VCT can stand in for at
  least part of the clinical evidence for an imaging device.
* **Model-based selection for proton therapy (Netherlands).** Each patient's CT gets
  both a photon and a proton plan. NTCP models predict the toxicity gain, and only
  patients above a threshold benefit are referred for protons (Langendijk et al., 2013).
  This is in effect a per-patient virtual trial that is used in routine reimbursement
  decisions.
* **Planning and dose-prediction challenges.** Open datasets such as OpenKBP
  (Babier et al., 2021) let deep-learning dose predictors be benchmarked on shared CT +
  contour inputs. The same predictors can generate achievable-dose estimates for
  thousands of virtual patients or protocol variants.
* **Oncology digital twins.** Imaging-calibrated mechanistic models (e.g. MRI-informed
  tumour growth and response) are being used to simulate other dosing schedules or
  regimens for an individual before treatment (Wu et al., 2022).

## 4. Credibility and regulation

A predictive VCT is only as trustworthy as its validated context of use:

* **ASME V&V 40-2018** and the **FDA guidance on assessing the credibility of
  computational modelling and simulation in medical device submissions (2023)** set out
  a risk-informed framework: define the question of interest and context of use, judge
  the model's influence and the consequence of a wrong decision, then scale
  verification, validation and uncertainty quantification to match.
* Viceconti et al. (2021) describe VV&UQ practice specific to in silico trials.
* **Key validation steps:** realism of the phantoms against real image statistics;
  agreement between simulated and real scanner output; agreement between the model
  observer and humans; and, most important, **agreement with an existing clinical trial**
  on a shared endpoint before the VCT is used to extrapolate.

## 5. Open problems

1. **Representativeness.** Phantom libraries and training cohorts under-represent some
   anatomies, ethnicities, implants and rare pathologies. A virtual trial can hide bias
   as easily as it can show it.
2. **Outcome-model uncertainty.** Errors compound along phantom → physics → reader/NTCP.
   Uncertainty should be propagated end to end, not just reported per component.
3. **Human factors.** Model observers do not capture reader fatigue, workflow or
   prevalence effects. Treatment-planner variability is often idealised away.
4. **Validation of generative patients.** There are no agreed metrics yet that show
   synthetic anatomy is "realistic enough" for a given endpoint.
5. **Hybrid designs.** The most promising near-term use is *augmenting* real trials
   (synthetic control arms, enrichment, protocol optimisation before a trial, sample-size
   planning) rather than replacing them.

## 6. Relevance to DoseVLM

DoseVLM predicts 3-D VMAT dose from CT, RTSTRUCT and RTPLAN, which makes it a natural
**intervention simulator** for radiotherapy virtual trials:

* **Fast virtual arms.** Changing the prescription, fractionation, arc set-up or OAR
  priorities in the plan text and re-predicting the dose gives a candidate dose
  distribution per arm in seconds, without a full TPS optimisation per patient.
* **Coupling to outcome models.** Predicted DVHs (D95, Dmean, D0.1cc) can go straight
  into NTCP/TCP models to estimate toxicity and control differences between arms. This
  mirrors the model-based selection approach.
* **Cohort scaling.** Run on large retrospective DICOM-RT archives, or on phantom and
  generative anatomies, it can estimate how a protocol change affects a whole population
  and its subgroups.
* **Caveats.** Predictions reflect the training cohort's planning style and are
  research-only (see README). For a credible VCT, the dose model's error (dose score, DVH
  score, gamma) should be propagated into the outcome endpoint as part of the
  uncertainty budget, and the context of use should be stated in V&V 40 terms.

## References

* Abadi E, Segars WP, Tsui BMW, et al. Virtual clinical trials in medical imaging: a
  review. *J Med Imaging.* 2020;7(4):042805.
* Babier A, Zhang B, Mahmood R, et al. OpenKBP: the open-access knowledge-based planning
  grand challenge and dataset. *Med Phys.* 2021;48(9):5549–5561.
* Badano A, Graff CG, Badal A, et al. Evaluation of digital breast tomosynthesis as
  replacement of full-field digital mammography using an in silico imaging trial.
  *JAMA Netw Open.* 2018;1(7):e185474.
* Langendijk JA, Lambin P, De Ruysscher D, et al. Selection of patients for radiotherapy
  with protons aiming at reduction of side effects: the model-based approach.
  *Radiother Oncol.* 2013;107(3):267–273.
* Segars WP, Sturgeon G, Mendonca S, Grimes J, Tsui BMW. 4D XCAT phantom for multimodality
  imaging research. *Med Phys.* 2010;37(9):4902–4915.
* Viceconti M, Pappalardo F, Rodriguez B, et al. In silico trials: verification,
  validation and uncertainty quantification of predictive models used in the regulatory
  evaluation of biomedical products. *Methods.* 2021;185:120–127.
* Wu C, Lorenzo G, Hormuth DA, et al. Integrating mechanism-based modeling with
  biomedical imaging to build practical digital twins for clinical oncology.
  *Biophys Rev.* 2022;3(2):021304.
* Yankeelov TE, Atuegwu N, Hormuth D, et al. Clinically relevant modeling of tumor growth
  and treatment response. *Sci Transl Med.* 2013;5(187):187ps9.
* ASME V&V 40-2018. Assessing credibility of computational modeling through verification
  and validation: application to medical devices.
* U.S. FDA. Assessing the credibility of computational modeling and simulation in medical
  device submissions. Guidance for industry and FDA staff, 2023.

*Bibliographic details were compiled from memory. Check volume and page numbers against
the original sources before formal citation.*
