"""One-line treatment recommendations per PlantVillage class.

Covers all 38 classes of the full PlantVillage dataset; the active 15-class
subset in data/data.yaml is a strict subset of these keys. General guidance
for a demo — not a substitute for local agronomic advice.
"""

TREATMENTS = {
    "Apple — Apple scab": "Apply captan or myclobutanil at green tip; rake and destroy fallen leaves to cut overwintering inoculum.",
    "Apple — Black rot": "Prune out cankers and mummified fruit; apply captan or thiophanate-methyl from petal fall onward.",
    "Apple — Cedar apple rust": "Remove nearby junipers where practical; spray myclobutanil from pink bud through second cover.",
    "Apple — healthy": "No disease detected — maintain regular scouting and a balanced spray schedule.",
    "Blueberry — healthy": "No disease detected — keep soil pH 4.5–5.5 and prune for airflow.",
    "Cherry — Powdery mildew": "Apply sulfur or myclobutanil at shuck fall; prune for canopy airflow and avoid excess nitrogen.",
    "Cherry — healthy": "No disease detected — continue routine monitoring after rain events.",
    "Corn — Cercospora / Gray leaf spot": "Rotate away from corn 1–2 years, use resistant hybrids; strobilurin fungicide at VT–R1 if pressure is high.",
    "Corn — Common rust": "Usually cosmetic in field corn; plant resistant hybrids, spray triazole fungicide only under heavy early infection.",
    "Corn — Northern leaf blight": "Choose hybrids with Ht resistance genes; apply fungicide at tasseling when lesions reach the upper canopy.",
    "Corn — healthy": "No disease detected — scout weekly from V8 through R2.",
    "Grape — Black rot": "Remove mummified berries, open the canopy; spray mancozeb or myclobutanil from bud break to bunch closure.",
    "Grape — Esca (Black Measles)": "No chemical cure — prune infected wood in dry weather, protect pruning wounds, and remove badly infected vines.",
    "Grape — Leaf blight (Isariopsis)": "Improve airflow and sanitation; protectant sprays (mancozeb, copper) used for downy mildew also control it.",
    "Grape — healthy": "No disease detected — keep the fruiting zone well ventilated.",
    "Orange — Huanglongbing (Citrus greening)": "No cure — remove infected trees, control Asian citrus psyllid vectors, and plant certified disease-free stock.",
    "Peach — Bacterial spot": "Plant tolerant cultivars; apply copper at leaf drop and oxytetracycline during the season; avoid windy, sandy sites.",
    "Peach — healthy": "No disease detected — maintain dormant copper sprays as preventive care.",
    "Bell pepper — Bacterial spot": "Use certified clean seed and resistant varieties; apply copper + mancozeb early; avoid overhead irrigation.",
    "Bell pepper — healthy": "No disease detected — water at the base and rotate out of solanaceous crops.",
    "Potato — Early blight": "Apply chlorothalonil or azoxystrobin at first sign; maintain plant vigor with balanced fertility and rotate 2+ years.",
    "Potato — Late blight": "Act fast — apply chlorothalonil or mefenoxam-based fungicide, destroy cull piles and volunteers, hill tubers well.",
    "Potato — healthy": "No disease detected — monitor closely during cool, wet spells.",
    "Raspberry — healthy": "No disease detected — thin canes for airflow and remove spent floricanes.",
    "Soybean — healthy": "No disease detected — continue scouting through pod fill.",
    "Squash — Powdery mildew": "Spray sulfur, potassium bicarbonate, or myclobutanil at first white patches; plant resistant varieties next season.",
    "Strawberry — Leaf scorch": "Renovate beds after harvest, remove old foliage; apply captan or copper during early bloom in wet years.",
    "Strawberry — healthy": "No disease detected — renew plantings every 3–4 years.",
    "Tomato — Bacterial spot": "Use clean transplants; spray copper + mancozeb weekly in wet weather; never work plants while wet.",
    "Tomato — Early blight": "Remove lower infected leaves, mulch the soil line, and spray chlorothalonil or copper every 7–10 days.",
    "Tomato — Late blight": "Destroy infected plants immediately; protect the rest with chlorothalonil or copper — spreads farm-to-farm fast.",
    "Tomato — Leaf mold": "A greenhouse problem — drop humidity below 85%, improve venting, and spray chlorothalonil if needed.",
    "Tomato — Septoria leaf spot": "Strip infected lower leaves, mulch, rotate 2 years; chlorothalonil or copper on a 7–10 day schedule.",
    "Tomato — Spider mites": "Spray insecticidal soap or horticultural oil on leaf undersides; avoid broad-spectrum insecticides that kill predators.",
    "Tomato — Target spot": "Improve airflow and reduce leaf wetness; rotate fungicide groups (chlorothalonil, azoxystrobin) to prevent resistance.",
    "Tomato — Yellow leaf curl virus": "No cure — remove infected plants, control whitefly vectors with reflective mulch and insecticidal soap.",
    "Tomato — Mosaic virus": "No cure — rogue out infected plants, disinfect tools with 10% bleach, wash hands; avoid tobacco handling.",
    "Tomato — healthy": "No disease detected — keep foliage dry and feed steadily.",
}


def treatment_for(class_name: str) -> str:
    return TREATMENTS.get(class_name, "No treatment note available for this class.")
