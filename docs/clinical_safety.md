# Clinical safety

Research use only — not validated for autonomous clinical diagnosis or patient management.

## Decisions a physician must make

The software can propose a box, a mask, and a benign/malignant research label. It cannot:

- declare image quality adequate without a reviewer
- pass a rejected box or mask into the report
- invent a BI-RADS descriptor
- convert a benign/malignant probability into BI-RADS 3, 4A, 4B, 4C, or 5
- assign BI-RADS 0 or 6 from the image
- sign the report

The API stores the reviewer id, the action, and the time. Set `REVIEWER_TOKEN` before starting it. Unsigned approvals are rejected. Uploaded file names are not stored. Do not upload identified clinical studies to this prototype. The dataset in this repository is the public BUSI teaching set, and even that set should be treated as sensitive research data.

## Uncertainty the software keeps

- Missing descriptors stay null.
- Low-confidence or absent detections stay unapproved until a person draws or accepts a box.
- Empty masks do not receive a surface distance or an orientation.
- If the supplied PDFs do not contain an answer, retrieval returns no supporting chunk rather than a fabricated citation. The 10-question check is a phrase test, not a proof of clinical grounding.
- The language model is not on the path that builds `StructuredStudyReport`.

## What the older Stage 6 panel still does

`app.py` can show heuristic lexicon sentences from radiomic thresholds. The rule-based category is off unless `reporting.show_rule_category` is true. Those sentences are not approved findings. Prefer the structured report, which leaves descriptor fields null until a qualified reviewer fills them from an annotation protocol.

## Display requirement

Any demo screen should show:

Research use only — not validated for autonomous clinical diagnosis or patient management.
