"""
Medical metadata, clinical feature definitions, reference ranges, and test cases.
"""

FEATURE_DEFINITIONS = {
    "radius_mean": {
        "label": "Mean Radius",
        "unit": "µm",
        "group": "mean",
        "description": "Mean distance from center to cell nucleus perimeter.",
        "normal_range": "10.0 - 15.0 µm",
        "clinical_significance": "Malignant nuclei are typically larger due to uncontrolled proliferation."
    },
    "texture_mean": {
        "label": "Mean Texture",
        "unit": "score",
        "group": "mean",
        "description": "Standard deviation of gray-scale values in digitized image.",
        "normal_range": "14.0 - 20.0",
        "clinical_significance": "Reflects chromatin coarseness; malignant cells show higher texture variation."
    },
    "perimeter_mean": {
        "label": "Mean Perimeter",
        "unit": "µm",
        "group": "mean",
        "description": "Mean nuclear boundary perimeter length.",
        "normal_range": "65.0 - 100.0 µm",
        "clinical_significance": "Directly correlates with tumor volume and nuclear enlargement."
    },
    "area_mean": {
        "label": "Mean Area",
        "unit": "µm²",
        "group": "mean",
        "description": "Mean nuclear cross-sectional area.",
        "normal_range": "300.0 - 700.0 µm²",
        "clinical_significance": "Significantly enlarged in invasive carcinomas."
    },
    "smoothness_mean": {
        "label": "Mean Smoothness",
        "unit": "ratio",
        "group": "mean",
        "description": "Local variation in radius lengths.",
        "normal_range": "0.08 - 0.11",
        "clinical_significance": "Measures micro-irregularities along the nuclear envelope."
    },
    "compactness_mean": {
        "label": "Mean Compactness",
        "unit": "ratio",
        "group": "mean",
        "description": "Perimeter² / Area - 1.0; measures circularity deviation.",
        "normal_range": "0.05 - 0.12",
        "clinical_significance": "Higher values indicate pleomorphic, non-circular cancerous cells."
    },
    "concavity_mean": {
        "label": "Mean Concavity",
        "unit": "ratio",
        "group": "mean",
        "description": "Severity of concave indentations along the contour.",
        "normal_range": "0.01 - 0.08",
        "clinical_significance": "Strong discriminator; malignant contours feature deep, sharp indentations."
    },
    "concave points_mean": {
        "label": "Mean Concave Points",
        "unit": "count/ratio",
        "group": "mean",
        "description": "Number of concave portions along the nuclear contour.",
        "normal_range": "0.01 - 0.05",
        "clinical_significance": "One of the top statistical indicators of cellular atypia."
    },
    "symmetry_mean": {
        "label": "Mean Symmetry",
        "unit": "ratio",
        "group": "mean",
        "description": "Relative balance of nuclear chromatin distribution.",
        "normal_range": "0.15 - 0.20",
        "clinical_significance": "Asymmetric nuclear shapes are common in malignant transformations."
    },
    "fractal_dimension_mean": {
        "label": "Mean Fractal Dimension",
        "unit": "dimension",
        "group": "mean",
        "description": "Coastline approximation - 1; boundary complexity.",
        "normal_range": "0.05 - 0.07",
        "clinical_significance": "Quantifies the irregular fractal nature of tumor borders."
    },

    # Standard Error (SE) features
    "radius_se": {
        "label": "Radius (SE)",
        "unit": "µm",
        "group": "se",
        "description": "Standard error for the radius measurement.",
        "normal_range": "0.15 - 0.45",
        "clinical_significance": "Higher variability across cells indicates cellular pleomorphism."
    },
    "texture_se": {
        "label": "Texture (SE)",
        "unit": "score",
        "group": "se",
        "description": "Standard error for grayscale texture across cells.",
        "normal_range": "0.70 - 1.50",
        "clinical_significance": "Fluctuations in chromatin density across sample cells."
    },
    "perimeter_se": {
        "label": "Perimeter (SE)",
        "unit": "µm",
        "group": "se",
        "description": "Standard error of perimeter length.",
        "normal_range": "1.0 - 3.0",
        "clinical_significance": "High variance indicates heterogeneous cell population."
    },
    "area_se": {
        "label": "Area (SE)",
        "unit": "µm²",
        "group": "se",
        "description": "Standard error of nuclear area.",
        "normal_range": "10.0 - 45.0",
        "clinical_significance": "Marks anisonucleosis (unequal cell nucleus sizes)."
    },
    "smoothness_se": {
        "label": "Smoothness (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of boundary smoothness.",
        "normal_range": "0.004 - 0.009",
        "clinical_significance": "Envelope variation between nuclei."
    },
    "compactness_se": {
        "label": "Compactness (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of nuclear compactness.",
        "normal_range": "0.01 - 0.035",
        "clinical_significance": "Structural heterogeneity."
    },
    "concavity_se": {
        "label": "Concavity (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of contour concavity.",
        "normal_range": "0.01 - 0.04",
        "clinical_significance": "Contour variation."
    },
    "concave points_se": {
        "label": "Concave Points (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of concave points.",
        "normal_range": "0.006 - 0.016",
        "clinical_significance": "Variability in contour indentations."
    },
    "symmetry_se": {
        "label": "Symmetry (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of symmetry.",
        "normal_range": "0.012 - 0.026",
        "clinical_significance": "Variability in symmetry."
    },
    "fractal_dimension_se": {
        "label": "Fractal Dimension (SE)",
        "unit": "ratio",
        "group": "se",
        "description": "Standard error of fractal dimension.",
        "normal_range": "0.002 - 0.005",
        "clinical_significance": "Boundary complexity variability."
    },

    # Worst features
    "radius_worst": {
        "label": "Worst Radius",
        "unit": "µm",
        "group": "worst",
        "description": "Largest ('worst') radius value observed in sample.",
        "normal_range": "11.0 - 18.0 µm",
        "clinical_significance": "Critical diagnostic factor; largest tumor cells drive malignant staging."
    },
    "texture_worst": {
        "label": "Worst Texture",
        "unit": "score",
        "group": "worst",
        "description": "Largest texture score observed.",
        "normal_range": "17.0 - 28.0",
        "clinical_significance": "Extreme chromatin clumping."
    },
    "perimeter_worst": {
        "label": "Worst Perimeter",
        "unit": "µm",
        "group": "worst",
        "description": "Largest perimeter measurement.",
        "normal_range": "75.0 - 120.0 µm",
        "clinical_significance": "Correlates directly with maximum tumor diameter."
    },
    "area_worst": {
        "label": "Worst Area",
        "unit": "µm²",
        "group": "worst",
        "description": "Largest nuclear area observed.",
        "normal_range": "400.0 - 1000.0 µm²",
        "clinical_significance": "Significant marker of high-grade malignancy."
    },
    "smoothness_worst": {
        "label": "Worst Smoothness",
        "unit": "ratio",
        "group": "worst",
        "description": "Most irregular local radius variation.",
        "normal_range": "0.10 - 0.15",
        "clinical_significance": "Severe membrane disruption."
    },
    "compactness_worst": {
        "label": "Worst Compactness",
        "unit": "ratio",
        "group": "worst",
        "description": "Highest compactness value observed.",
        "normal_range": "0.10 - 0.30",
        "clinical_significance": "High values signal aggressive cell distortion."
    },
    "concavity_worst": {
        "label": "Worst Concavity",
        "unit": "ratio",
        "group": "worst",
        "description": "Most severe concave indentations.",
        "normal_range": "0.05 - 0.35",
        "clinical_significance": "Top diagnostic indicator for malignant cell invagination."
    },
    "concave points_worst": {
        "label": "Worst Concave Points",
        "unit": "count/ratio",
        "group": "worst",
        "description": "Largest number of concave points.",
        "normal_range": "0.03 - 0.12",
        "clinical_significance": "Highly influential feature in Wisconsin diagnostic dataset."
    },
    "symmetry_worst": {
        "label": "Worst Symmetry",
        "unit": "ratio",
        "group": "worst",
        "description": "Highest asymmetry score.",
        "normal_range": "0.20 - 0.35",
        "clinical_significance": "Gross nuclear asymmetry."
    },
    "fractal_dimension_worst": {
        "label": "Worst Fractal Dimension",
        "unit": "dimension",
        "group": "worst",
        "description": "Highest boundary complexity measurement.",
        "normal_range": "0.06 - 0.09",
        "clinical_significance": "High micro-lobulation."
    }
}

CLINICAL_PRESET_CASES = [
    {
        "id": "CASE-BENIGN-01",
        "name": "Case 1: Typical Benign Lesion (Fibroadenoma)",
        "expected": "Benign",
        "summary": "Regular nuclear margins, small uniform nuclei, low concavity. Consistent with benign breast histology.",
        "data": {
            "radius_mean": 13.54,
            "texture_mean": 14.36,
            "perimeter_mean": 87.46,
            "area_mean": 566.3,
            "smoothness_mean": 0.09779,
            "compactness_mean": 0.08129,
            "concavity_mean": 0.06664,
            "concave points_mean": 0.04781,
            "symmetry_mean": 0.1885,
            "fractal_dimension_mean": 0.05766,
            "radius_se": 0.2699,
            "texture_se": 0.7886,
            "perimeter_se": 2.058,
            "area_se": 23.56,
            "smoothness_se": 0.008462,
            "compactness_se": 0.0146,
            "concavity_se": 0.02387,
            "concave points_se": 0.01315,
            "symmetry_se": 0.0198,
            "fractal_dimension_se": 0.0023,
            "radius_worst": 15.11,
            "texture_worst": 19.26,
            "perimeter_worst": 99.7,
            "area_worst": 711.2,
            "smoothness_worst": 0.144,
            "compactness_worst": 0.1773,
            "concavity_worst": 0.239,
            "concave points_worst": 0.1288,
            "symmetry_worst": 0.2977,
            "fractal_dimension_worst": 0.07259
        }
    },
    {
        "id": "CASE-MALIGNANT-02",
        "name": "Case 2: Invasive Ductal Carcinoma (High Grade)",
        "expected": "Malignant",
        "summary": "Marked anisonucleosis, prominent nuclear pleomorphism, high concavity, and extensive concave indentations.",
        "data": {
            "radius_mean": 17.99,
            "texture_mean": 10.38,
            "perimeter_mean": 122.8,
            "area_mean": 1001.0,
            "smoothness_mean": 0.1184,
            "compactness_mean": 0.2776,
            "concavity_mean": 0.3001,
            "concave points_mean": 0.1471,
            "symmetry_mean": 0.2419,
            "fractal_dimension_mean": 0.07871,
            "radius_se": 1.095,
            "texture_se": 0.9053,
            "perimeter_se": 8.589,
            "area_se": 153.4,
            "smoothness_se": 0.006399,
            "compactness_se": 0.04904,
            "concavity_se": 0.05373,
            "concave points_se": 0.01587,
            "symmetry_se": 0.03003,
            "fractal_dimension_se": 0.006193,
            "radius_worst": 25.38,
            "texture_worst": 17.33,
            "perimeter_worst": 184.6,
            "area_worst": 2019.0,
            "smoothness_worst": 0.1622,
            "compactness_worst": 0.6656,
            "concavity_worst": 0.7119,
            "concave points_worst": 0.2654,
            "symmetry_worst": 0.4601,
            "fractal_dimension_worst": 0.1189
        }
    },
    {
        "id": "CASE-BORDERLINE-03",
        "name": "Case 3: Borderline / Atypical Cell Feature (Indeterminate)",
        "expected": "Indeterminate / Pathologist Review Advised",
        "summary": "Intermediate nuclear enlargement with borderline concave points (~49.4% malignancy prob). Represents a challenging gray-zone case where AI flags uncertainty.",
        "data": {
            "radius_mean": 13.8,
            "texture_mean": 15.79,
            "perimeter_mean": 90.43,
            "area_mean": 584.1,
            "smoothness_mean": 0.1007,
            "compactness_mean": 0.128,
            "concavity_mean": 0.07789,
            "concave points_mean": 0.05069,
            "symmetry_mean": 0.1662,
            "fractal_dimension_mean": 0.06566,
            "radius_se": 0.2787,
            "texture_se": 0.6205,
            "perimeter_se": 1.957,
            "area_se": 23.35,
            "smoothness_se": 0.004717,
            "compactness_se": 0.02065,
            "concavity_se": 0.01759,
            "concave points_se": 0.009206,
            "symmetry_se": 0.0122,
            "fractal_dimension_se": 0.00313,
            "radius_worst": 16.57,
            "texture_worst": 20.86,
            "perimeter_worst": 110.3,
            "area_worst": 812.4,
            "smoothness_worst": 0.1411,
            "compactness_worst": 0.3542,
            "concavity_worst": 0.2779,
            "concave points_worst": 0.1383,
            "symmetry_worst": 0.2589,
            "fractal_dimension_worst": 0.103
        }
    }
]


def classify_risk(malignant_probability):
    """
    Clinically-grounded risk categorization for decision support.
    """
    prob_pct = round(malignant_probability * 100, 2)
    
    if malignant_probability < 0.25:
        category = "Low Risk"
        badge = "success"
        recommendation = (
            "Cytological characteristics strongly favor a benign diagnosis. "
            "Recommend routine surveillance per age-appropriate screening guidelines."
        )
        alert_level = "info"
    elif malignant_probability < 0.60:
        category = "Indeterminate / Borderline Risk"
        badge = "warning"
        recommendation = (
            "Cellular morphology exhibits intermediate atypia near decision boundaries. "
            "AI uncertainty is elevated. Pathologist microscopic review and repeat FNA or core needle biopsy advised."
        )
        alert_level = "warning"
    else:
        category = "High Risk of Malignancy"
        badge = "danger"
        recommendation = (
            "Cytomorphometric profile indicates significant malignant cellular features. "
            "Prompt histological confirmation (core biopsy), radiological correlation (BI-RADS), and surgical oncology consultation recommended."
        )
        alert_level = "danger"

    return {
        "malignancy_probability": float(malignant_probability),
        "malignancy_percentage": prob_pct,
        "benign_percentage": round(100 - prob_pct, 2),
        "risk_category": category,
        "badge": badge,
        "recommendation": recommendation,
        "alert_level": alert_level,
        "requires_urgent_review": bool(malignant_probability >= 0.60),
        "is_borderline": bool(0.25 <= malignant_probability < 0.60)
    }
