"""
Agronomic Knowledge Base for Plant Nutrient Deficiency Diagnostic System.
Provides symptoms, causes, corrective actions, and multi-lingual guidance (English & Urdu).
"""

NUTRIENT_KNOWLEDGE_BASE = {
    "Healthy": {
        "title": "Healthy Leaf",
        "title_ur": "صحت مند پودا",
        "symptoms": [
            "Uniform vibrant green coloration across leaf blade and veins.",
            "Turgid and vigorous leaf structure with no necrotic spots or margin burns.",
            "Normal growth rate and optimal leaf surface area."
        ],
        "symptoms_ur": [
            "پتے کی پوری سطح پر یکساں اور چمکدار سبز رنگت۔",
            "پتے کی مضبوط بناوٹ، بغیر کسی داغ دھبے یا سوکھے کناروں کے۔",
            "معمول کی صحت مند نشوونما۔"
        ],
        "causes": [
            "Balanced soil nutrients, optimal pH, adequate moisture, and proper sunlight."
        ],
        "remedies": [
            "Maintain standard irrigation and fertilization schedules.",
            "Conduct routine soil health checks every 6-12 months.",
            "Practice integrated pest and disease management to sustain vigor."
        ],
        "fertilizer_recommendation": "Standard maintenance dose as per local agricultural extension guidelines.",
        "soil_factors": "Maintain optimal root-zone soil pH between 6.0 and 7.0."
    },
    "Nitrogen (N)": {
        "title": "Nitrogen (N) Deficiency",
        "title_ur": "نائٹروجن (N) کی کمی",
        "symptoms": [
            "General chlorosis (pale green to yellowing) starting on older, lower leaves.",
            "Gradual spread upward to younger leaves as nitrogen is translocated.",
            "Stunted plant growth, thin spindly stems, and reduced leaf size."
        ],
        "symptoms_ur": [
            "نچلے اور پرانے پتوں سے شروع ہونے والی عمومی زردی (پیلے پتے)۔",
            "وقت کے ساتھ ساتھ اوپر والے نئے پتوں کا بھی پیلا پڑ جانا۔",
            "پودے کی نشوونما کا رک جانا اور تنے کا پتلا و کمزور رہ جانا۔"
        ],
        "causes": [
            "Leaching in sandy or high-rainfall soils, insufficient organic matter, or waterlogged anaerobic root conditions."
        ],
        "remedies": [
            "Apply fast-acting nitrogen fertilizer (Urea, Ammonium Nitrate, or Calcium Ammonium Nitrate).",
            "Incorporate well-composted farmyard manure or organic vermicompost into the root zone.",
            "For rapid relief, apply a foliar spray of 1-2% urea solution in the early morning or evening."
        ],
        "fertilizer_recommendation": "Apply Urea (46-0-0) or Ammonium Sulphate according to crop-specific nitrogen requirements.",
        "soil_factors": "Check for soil compaction or excessive leaching. Nitrogen uptake is best between pH 6.0 - 7.5."
    },
    "Phosphorus (P)": {
        "title": "Phosphorus (P) Deficiency",
        "title_ur": "فاسفورس (P) کی کمی",
        "symptoms": [
            "Abnormally dark green leaves, developing purplish or bronze pigmentation along veins and margins.",
            "Severely stunted root system and delayed maturity or flowering.",
            "Erect, narrow leaves with necrotic margins in prolonged deficiency."
        ],
        "symptoms_ur": [
            "پتوں کا ضرورت سے زیادہ گہرا سبز ہو جانا اور کناروں پر جامنی یا سرخ رنگت نمودار ہونا۔",
            "جڑوں کی نشوونما کا بری طرح متاثر ہونا اور پھول/پھل دیر سے آنا۔",
            "پتوں کا تنگ اور سیدھا کھڑا رہنا۔"
        ],
        "causes": [
            "Strong phosphorus fixation in highly acidic (pH < 5.5) or strongly alkaline/calcareous soils (pH > 7.5); cold root zone temperatures."
        ],
        "remedies": [
            "Apply Diammonium Phosphate (DAP), Single Super Phosphate (SSP), or Triple Super Phosphate (TSP).",
            "Inoculate soil with Phosphate Solubilizing Bacteria (PSB) to unlock fixed native phosphorus.",
            "Foliar application of Mono-Potassium Phosphate (MKP 0-52-34) for rapid crop response."
        ],
        "fertilizer_recommendation": "Apply DAP (18-46-0) or SSP banded close to the root zone at planting.",
        "soil_factors": "Phosphorus availability peaks at pH 6.2 - 6.8. In calcareous alkaline soils, use acidic P formulations."
    },
    "Potassium (K)": {
        "title": "Potassium (K) Deficiency",
        "title_ur": "پوٹاشیم (K) کی کمی",
        "symptoms": [
            "Marginal chlorosis quickly followed by necrosis (tip and edge burn/scorching) on older leaves.",
            "Curling or cupping of leaf edges, weak lodging stems, and poor drought tolerance.",
            "Interveinal chlorosis with dark necrotic spots in severe cases."
        ],
        "symptoms_ur": [
            "پرانے پتوں کے کناروں اور سروں کا جل جانا یا بھورا سوکھ جانا (کنارے جلنا)۔",
            "پتوں کا کناروں سے مڑ جانا اور تنے کی کمزوری جس سے پودا گر سکتا ہے۔",
            "پودے میں خشک سالی اور بیماریوں کے خلاف قوت مدافعت کم ہونا۔"
        ],
        "causes": [
            "Low native potassium reserves in sandy/leached soils, excess magnesium or calcium competing for root uptake."
        ],
        "remedies": [
            "Apply Muriate of Potash (MOP / KCl) or Sulphate of Potash (SOP / Potassium Sulphate).",
            "Apply foliar Potassium Nitrate (13-0-45) at 1-2% concentration for instant leaf uptake.",
            "Mulch the soil surface with organic crop residues rich in potassium (e.g., banana pseudo-stem compost or ash)."
        ],
        "fertilizer_recommendation": "Apply Potassium Sulphate (0-0-50) or Muriate of Potash (0-0-60).",
        "soil_factors": "Potassium uptake is optimal at pH 6.0 - 7.5. Avoid excessive liming which inhibits K uptake."
    },
    "Iron (Fe)": {
        "title": "Iron (Fe) Deficiency (Iron Chlorosis)",
        "title_ur": "آئرن / لوہے (Fe) کی کمی",
        "symptoms": [
            "Distinct interveinal chlorosis on the youngest emerging leaves (veins remain distinctly green while the blade turns light yellow or ivory white).",
            "Leaves turn bleached white in severe deficiencies with necrotic margin dieback."
        ],
        "symptoms_ur": [
            "نئے نکلنے والے اوپر کے پتوں کی رگوں کے درمیان پیلا پن جبکہ رگیں بالکل سبز رہتی ہیں۔",
            "شدید حالت میں پتے تقریباً سفید ہو جاتے ہیں اور خشک ہو کر جھڑنے لگتے ہیں۔"
        ],
        "causes": [
            "High soil pH (> 7.5), high calcium carbonate (calcareous soils), poor aeration, or excessive soil moisture which immobilizes iron."
        ],
        "remedies": [
            "Foliar spray with 0.5% - 1.0% Chelated Iron (Fe-EDTA or Fe-EDDHA for alkaline soils).",
            "Apply elemental sulphur or acidic organic compost to lower soil pH in the rhizosphere.",
            "Improve drainage to enhance soil oxygen and root iron reduction."
        ],
        "fertilizer_recommendation": "Foliar application of Fe-EDTA (12% Fe) or soil application of Fe-EDDHA (6% Fe) in calcareous soils.",
        "soil_factors": "Iron deficiency is strongly pH-driven. It rarely occurs below pH 6.5; common in alkaline soils (pH > 7.8)."
    },
    "Magnesium (Mg)": {
        "title": "Magnesium (Mg) Deficiency",
        "title_ur": "میگنیشیم (Mg) کی کمی",
        "symptoms": [
            "Interveinal chlorosis beginning on older, lower leaves (veins remain green with broad yellow zones between veins).",
            "In some species (e.g. cotton, brassica), leaves develop reddish, purplish, or bronze pigmentation.",
            "Brittle leaves that curl upward along the margins."
        ],
        "symptoms_ur": [
            "پرانے اور نچلے پتوں میں رگوں کے درمیانی حصے کا پیلا یا سرخی مائل ہونا جبکہ موٹی رگیں سبز رہتی ہیں۔",
            "پتوں کا سخت اور ٹوٹنے والا بن جانا اور کناروں سے اوپر کی طرف مڑنا۔"
        ],
        "causes": [
            "Acidic sandy soils subject to high leaching, or excessive potassium/ammonium fertilizer application blocking Mg absorption."
        ],
        "remedies": [
            "Apply Magnesium Sulphate (Epsom Salt, MgSO4) as a soil amendment or foliar spray (1-2%).",
            "Apply Dolomitic Limestone (CaCO3 + MgCO3) on acidic soils to simultaneously neutralize acidity and supply magnesium.",
            "Balance potassium and ammonium fertilizer doses to avoid cation antagonism."
        ],
        "fertilizer_recommendation": "Soil application of Dolomite or Epsom Salt (Magnesium Sulphate heptahydrate).",
        "soil_factors": "Most prevalent in acidic soils (pH < 5.5). Correcting soil pH significantly improves availability."
    },
    "Calcium (Ca)": {
        "title": "Calcium (Ca) Deficiency",
        "title_ur": "کیلشیم (Ca) کی کمی",
        "symptoms": [
            "Deformed, hooked, or cupped youngest leaves with necrotic leaf margins and growing tip death (terminal bud dieback).",
            "Blossom end rot in fruiting crops (e.g. tomatoes, peppers) and bitter pit in fruit.",
            "Poor root development with stunted, gelatinous root tips."
        ],
        "symptoms_ur": [
            "نئے نکلنے والے پتوں اور اگنے والے سروں کی خرابی، کٹا پھٹا پن اور سروں کا سوکھنا۔",
            "پھل کی نچلی سطح کا سیاہ ہو کر سڑ جانا (جیسے ٹماٹر میں بلوسم اینڈ راٹ)۔",
            "جڑوں کی نشوونما کا شدید متاثر ہونا۔"
        ],
        "causes": [
            "Acidic sandy soils, irregular watering (calcium moves solely via transpiration stream), high humidity hindering transpiration, or high sodium/potassium."
        ],
        "remedies": [
            "Apply Calcium Nitrate as a quick-release fertilizer or foliar spray (0.5 - 1%).",
            "Apply Agricultural Gypsum (Calcium Sulphate) on alkaline/sodic soils, or Agricultural Lime on acidic soils.",
            "Maintain consistent soil moisture levels to ensure uninterrupted transpiration flow."
        ],
        "fertilizer_recommendation": "Foliar or fertigation with Calcium Nitrate (15.5-0-0 + 19% Ca) or soil application of Gypsum.",
        "soil_factors": "Calcareous soils have plenty of Ca; sandy and strongly acidic soils (pH < 5.2) lack available calcium."
    },
    "Sulphur (S)": {
        "title": "Sulphur (S) Deficiency",
        "title_ur": "سلفر / گندھک (S) کی کمی",
        "symptoms": [
            "Uniform pale green to yellow leaves beginning primarily on younger upper leaves (unlike nitrogen which starts on older leaves).",
            "Stunted plant growth, delayed maturity, and reduced oil/protein content in oilseeds and legumes.",
            "Stiff, brittle stems."
        ],
        "symptoms_ur": [
            "اوپر والے نئے پتوں کا ہلکا سبز یا پیلا پڑ جانا (نائٹروجن کے برعکس جو نیچے سے شروع ہوتی ہے)۔",
            "پودے کا قد چھوٹا رہنا اور بیجوں میں تیل اور پروٹین کی کمی۔"
        ],
        "causes": [
            "Leaching in sandy soils, low organic matter, and reduced atmospheric deposition in modern agriculture."
        ],
        "remedies": [
            "Apply Ammonium Sulphate, Single Super Phosphate (SSP contains ~12% S), or Gypsum.",
            "Foliar application of micronized elemental sulphur or water-soluble sulphate fertilizers.",
            "Incorporate organic crop residues and farm manure."
        ],
        "fertilizer_recommendation": "Apply Ammonium Sulphate (21-0-0-24S) or Gypsum.",
        "soil_factors": "Sulphate (SO4(2-)) is easily leached in high-rainfall sandy soils. Best absorbed at pH 6.0 - 7.5."
    },
    "Zinc (Zn)": {
        "title": "Zinc (Zn) Deficiency",
        "title_ur": "زنک (Zn) کی کمی",
        "symptoms": [
            "Broad band interveinal chlorosis (white buds or bleached patches on mid-leaf blade).",
            "Shortened internodes leading to rosetting ('little leaf' symptom) and stunted growth.",
            "Bronzing and mottled yellowing on older and middle leaves."
        ],
        "symptoms_ur": [
            "پتوں کے درمیانی حصے پر چوڑی سفید یا پیلی پٹیاں نمودار ہونا (وائٹ بڈ)۔",
            "پتوں کا سائز چھوٹا رہ جانا اور گھچے کی شکل میں اکٹھے ہو جانا (چھوٹے پتے)۔",
            "پودے کے قد کا ٹھنگنا رہ جانا۔"
        ],
        "causes": [
            "Calcareous soils with high pH (> 7.5), high soil phosphorus levels antagonizing zinc absorption, waterlogged conditions."
        ],
        "remedies": [
            "Apply Zinc Sulphate (21% or 33%) as basal soil application.",
            "Foliar spray with 0.5% Zinc Sulphate + 0.25% Lime or Chelated Zinc (Zn-EDTA).",
            "Avoid over-application of phosphorus fertilizers which induces zinc lockup."
        ],
        "fertilizer_recommendation": "Foliar application of Zn-EDTA (12%) or soil application of Zinc Sulphate.",
        "soil_factors": "Zinc availability drops 100-fold for each unit increase in soil pH above 7.0."
    }
}

DISCLAIMER_TEXT = (
    "Agricultural Diagnostic Disclaimer: The predictions and recommendations provided by this AI system "
    "are for preliminary screening and visual decision support only. Visual symptoms may overlap with "
    "pest infestations, viral/fungal diseases, water stress, or salinity. Soil and plant-tissue laboratory "
    "testing should always be conducted before applying large amounts of chemical fertilizers."
)

DISCLAIMER_TEXT_UR = (
    "زرعی تشخیصی انتباہ: اس مصنوعی ذہانت کے نظام کی فراہم کردہ پیش گوئیاں صرف ابتدائی رہنمائی اور معائنے کے لیے ہیں۔ "
    "پتوں کی ظاہری علامات کیڑوں، بیماریوں، پانی کی کمی یا مٹی کی نمکیات سے بھی مشابہ ہو سکتی ہیں۔ "
    "کھاد کے بڑے پیمانے پر استعمال سے پہلے ہمیشہ مٹی اور پودوں کے لیبارٹری ٹیسٹ کروائیں۔"
)
