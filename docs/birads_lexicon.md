# BI-RADS ultrasound lexicon used by this project

Primary sources on disk:

- ACR BI-RADS Atlas, fifth edition, quick reference card, `dataset/BIRADS Reference Card 1.pdf`. Ultrasound terms are on PDF page 1. Assessment categories are on PDF page 2. Card date 07.15. The full Atlas is not in the repository.
- Lee J. Practical and illustrated summary of updated BI-RADS for ultrasonography. Ultrasonography. 2017;36(1):71–81. `dataset/usg-16034 (1).pdf`. PDF page `n` is printed page `n + 70`.

This file lists terms from those documents. It does not add categories. Nothing here is a label for a BUSI image.

## Tissue composition

Screening ultrasound, card page 1:

- Homogeneous background echotexture – fat
- Homogeneous background echotexture – fibroglandular
- Heterogeneous background echotexture

Lee, printed p. 73, says the dominant tissue composition should be mentioned on a screening ultrasound and should correlate with mammography when mammography was performed.

## Mass shape

Card page 1 and Lee, printed pp. 73–75: oval, round, irregular.

Lee, printed p. 73: oval is elliptical or egg-shaped and may have up to two or three gentle lobulations; round is spherical or ball-shaped; irregular is neither round nor oval.

## Orientation

Lee, printed p. 76: orientation is the lesion’s long axis relative to the skin. The choices are parallel and not parallel. Parallel replaces the spoken phrase “wider-than-tall”; not parallel replaces “taller-than-wide.” By definition, a round mass is neither parallel nor not parallel.

A mask without a skin line is not enough to assign this. `bus_research/geometry.py` therefore leaves orientation empty.

## Margin

Card page 1. The first split is circumscribed or not circumscribed. Not-circumscribed subtypes on the card: indistinct, angular, microlobulated, spiculated.

Lee, printed pp. 75–76: if the margin is not circumscribed, the subtypes may be used in combination. A circumscribed margin suggests benignity and is not specific for it. The subtypes suggest “not benign” and do not guarantee malignancy.

## Echo pattern

Card page 1: anechoic, hyperechoic, complex cystic and solid, hypoechoic, isoechoic, heterogeneous.

Lee, printed p. 76: fat is the reference echotexture.

## Posterior features

Card page 1: no posterior features, enhancement, shadowing, combined pattern.

Lee, printed pp. 76–78: these describe attenuation relative to acoustic transmission. Enhancement is not specific for a benign diagnosis. Shadowing is associated with fibrosis, with or without carcinoma.

## Calcifications

Card page 1: calcifications in a mass, calcifications outside of a mass, intraductal calcifications.

Lee’s figure discussion, printed pp. 79–80, shows ultrasound calcifications together with the mammogram. An echogenic speck on a B-mode still is not recorded here as a calcification. This dataset has no mammogram and no calcification label.

## Associated features

Card page 1: architectural distortion, duct changes, skin changes (skin thickening, skin retraction), edema, vascularity (absent, internal vascularity, vessels in rim), elasticity (soft, intermediate, hard).

Vascularity and elasticity are not visible on a grayscale still. `bus_research/birads.py` marks them unavailable for that reason.

## Special cases

Card page 1: simple cyst, clustered microcysts, complicated cyst, mass in or on skin, foreign body including implants, intramammary lymph node, axillary lymph node, vascular abnormalities (AVMs / pseudoaneurysms, Mondor disease), postsurgical fluid collection, fat necrosis.

No special-case labels are in the image folders, so none of these are predicted.

## Assessment categories

Card page 2:

| Category | Wording on the card |
| --- | --- |
| 0 | Ultrasound and MRI: Incomplete – Need Additional Imaging Evaluation |
| 1 | Negative |
| 2 | Benign |
| 3 | Probably Benign |
| 4 | Suspicious. Ultrasound subcategories 4A, 4B, 4C |
| 4A | Low suspicion for malignancy |
| 4B | Moderate suspicion for malignancy |
| 4C | High suspicion for malignancy |
| 5 | Highly Suggestive of Malignancy |
| 6 | Known Biopsy-Proven Malignancy |

Lee, printed pp. 79–80, quotes these likelihood ranges from the Atlas table and says to refer to that table:

- Category 3: >0% to ≤2%
- Category 4 overall: >2% and <95%, with 4A >2% to ≤10%, 4B >10% to ≤50%, 4C >50% to <95%
- Category 5: ≥95%

Those ranges are reference text. They are not a function of a CNN probability. Categories 0 and 6 are not points on that scale. Category 0 is an incomplete assessment. Category 6 is a known biopsy result.
