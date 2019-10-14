import numpy as np
import pandas as pd

"""
    Inputs:
        wine_id: Wine ID assigned
        color: White or Red Wine
        alc: Alcohol (v/v%)
        red_sug: Reducing Sugar (g/100mL)
        hue: Hue
        hunt_a: Hunter A
        hunt_b: Hunter B
        hunt_l: Hunter L
        vol_acid: Volatile Acidity (g/100mL)
        titra_acid: Titratable Acidity (g/100mL)
        obrix_gal: OBrix (assuming this is the value from GAL Analysis)
        juice: Juice pH
        ph: pH
        va: VA (mg/L)
        obrix_must: OBrix (assuming this is the value from Initial Must Analysis)
        tot_sulfur_must: Total Sulfur (mg/L) (assuming this is the value from Initial Must Analysis)
        free_sulfur: Free Sulfur (mg/L)
        mol_sulfur: Molecular Sulfur (mg/L)
        copper: Copper (mg/L)
        tot_sulfur_chem: Total Sulfur (mg/L) (assuming this is the value from Basic Wine Chemistry Analysis)
        mal_acid: Malic Acid (mg/L)
        ibmp_gal: IBMP (ppt) (assuming this is the value from GAL Analysis)
        b_dama_gal: b-damascenone (ppb) (assuming this is the value from GAL Analysis)
        lin_gal: Linalool (ppb) (assuming this is the value from GAL Analysis)
        nerol: Nerol (ppb)
        ta: TA (g/100mL)
        nh3: NH3 (mg/L)
        nopa: NOPA (ppm)
        yan: YAN (ppm)
        hexanol1: 1-Hexanol (ppb)
        methyl_ace2: 2-Methylbutyl acetat (ppb)
        phenyl_ace2: 2-Phenylethyl acetate (ppb)
        b_dama_aroma: b-damascenone (ppb) (assuming this is the value from Wine Aroma Analysis)
        eth_but: Ethyl butanoate (ppb)
        eth_hex: Ethyl hexanoate (ppb)
        eth_isobut: Ethyl isobutyrate (ppb)
        eth_isoval: Ethyl isovalerate (ppb)
        eth_oct: Ethyl octanoate (ppb)
        iso_ace: Isoamyl acetate (ppb)
        lin_aroma: Linalool (ppb) (assuming this is the value from Wine Aroma Analysis)
        ibmp_aroma: IBMP (ppt) (assuming this is the value from Wine Aroma Analysis)
        intensity: Intensity
        caft_acid: Caftaric Acid (mg/L)
        catechin: Catechin (mg/L)
        grp: GRP (mg/L)

    Outputs:
        CSV file containing ground truth data for all wine samples

"""

    # 6 rows x 45 columns
    # values plugged in manually :( , excel sheet format not easy to automate data extraction

wine_id           = []
color             = ['white', 'white', 'white', 'white', 'white', 'white']
alc               = []
red_sug           = []
hue               = []
hunt_a            = []
hunt_b            = []
hunt_l            = []
vol_acid          = []
titra_acid        = []
obrix_gal         = []
juice             = []
ph                = []
va                = []
obrix_must        = []
tot_sulfur_must   = []
free_sulfur       = []
mol_sulfur        = []
copper            = []
tot_sulfur_chem   = []
mal_acid          = []
ibmp_gal          = []
b_dama_gal        = []
lin_gal           = []
nerol             = []
ta                = []
nh3               = []
nopa              = []
yan               = []
hexanol1          = []
methyl_ace2       = []
phenyl_ace2       = []
b_dama_aroma      = []
eth_but           = []
eth_hex           = []
eth_isobut        = []
eth_isoval        = []
eth_oct           = []
iso_ace           = []
lin_aroma         = []
ibmp_aroma        = []
intensity         = []
caft_acid         = []
catechin          = []
grp               = []

properties = [alc, red_sug, hue, hunt_a, hunt_b, hunt_l, vol_acid,
    titra_acid, obrix_gal, juice, ph, va, obrix_must, tot_sulfur_must, free_sulfur,
    mol_sulfur, copper, tot_sulfur_chem, mal_acid, ibmp_gal, b_dama_gal, lin_gal, nerol,
    ta , nh3, nopa, yan, hexanol1, methyl_ace2, phenyl_ace2, b_dama_aroma, eth_but, eth_hex,
    eth_isobut, eth_isoval, eth_oct, iso_ace, lin_aroma, ibmp_aroma, intensity, caft_acid, catechin, grp]

# Data Cleaning (all string values -> zero)
for prop in properties:
    for i in range(len(prop)):
        if(type(prop[i]) is str):
            prop[i] = 0.0

# Storing in DataFrame
data_dict = {
    'Wine ID': wine_id,
    'Color': color,
    'Alcohol (v/v%)': alc,
    'Reducing Sugar (g/100mL)': red_sug,
    'Hue': hue,
    'Hunter A': hunt_a,
    'Hunter B': hunt_b,
    'Hunter L': hunt_l,
    'Volatile Acidity (g/100mL)': vol_acid,
    'Titratable Acidity (g/100mL)': titra_acid,
    'OBrix (GAL)': obrix_gal,
    'Juice pH': juice,
    'pH': ph,
    'VA (mg/L)': va,
    'OBrix (Must)': obrix_must,
    'Total Sulfur (mg/L) (Must)': tot_sulfur_must,
    'Free Sulfur (mg/L)': free_sulfur,
    'Molecular Sulfur (mg/L)': mol_sulfur,
    'Copper (mg/L)': copper,
    'Total Sulfur (mg/L) (Chemistry)': tot_sulfur_chem,
    'Malic Acid (mg/L)': mal_acid,
    'IBMP (ppt) (GAL)': ibmp_gal,
    'b-damascenone (ppb) (GAL)': b_dama_gal,
    'Linalool (ppb) (GAL)': lin_gal,
    'Nerol (ppb)': nerol,
    'TA (g/100mL)': ta,
    'NH3 (mg/L)': nh3,
    'NOPA (ppm)': nopa,
    'YAN (ppm)': yan,
    '1-Hexanol (ppb)': hexanol1,
    '2-Methylbutyl acetat (ppb)': methyl_ace2,
    '2-Phenylethyl acetate (ppb)': phenyl_ace2,
    'b-damascenone (ppb) (Aroma)': b_dama_aroma,
    'Ethyl butanoate (ppb)': eth_but,
    'Ethyl hexanoate (ppb)': eth_hex,
    'Ethyl isobutyrate (ppb)': eth_isobut,
    'Ethyl isovalerate (ppb)': eth_isoval,
    'Ethyl octanoate (ppb)': eth_oct,
    'Isoamyl acetate (ppb)': iso_ace,
    'Linalool (ppb) (Aroma)': lin_aroma,
    'IBMP (ppt) (Aroma)': ibmp_aroma,
    'Intensity': intensity,
    'Caftaric Acid (mg/L)': caft_acid,
    'Catechin (mg/L)': catechin,
    'GRP (mg/L)': grp
}

df = pd.DataFrame(data_dict) # python dictionary -> pandas dataframe
cols = ['Wine ID', 'Color', 'Alcohol (v/v%)', 'Reducing Sugar (g/100mL)', 'Hue',
        'Hunter A', 'Hunter B', 'Hunter L', 'Volatile Acidity (g/100mL)', 'Titratable Acidity (g/100mL)',
        'OBrix (GAL)', 'Juice pH', 'pH', 'VA (mg/L)', 'OBrix (Must)', 'Total Sulfur (mg/L) (Must)',
        'Free Sulfur (mg/L)', 'Molecular Sulfur (mg/L)', 'Copper (mg/L)', 'Total Sulfur (mg/L) (Chemistry)',
        'Malic Acid (mg/L)', 'IBMP (ppt) (GAL)', 'b-damascenone (ppb) (GAL)', 'Linalool (ppb) (GAL)',
        'Nerol (ppb)', 'TA (g/100mL)', 'NH3 (mg/L)', 'NOPA (ppm)', 'YAN (ppm)', '1-Hexanol (ppb)',
        '2-Methylbutyl acetat (ppb)', '2-Phenylethyl acetate (ppb)', 'b-damascenone (ppb) (Aroma)',
        'Ethyl butanoate (ppb)', 'Ethyl hexanoate (ppb)', 'Ethyl isobutyrate (ppb)', 'Ethyl isovalerate (ppb)',
        'Ethyl octanoate (ppb)', 'Isoamyl acetate (ppb)', 'Linalool (ppb) (Aroma)', 'IBMP (ppt) (Aroma)',
        'Intensity', 'Caftaric Acid (mg/L)', 'Catechin (mg/L)', 'GRP (mg/L)']
df = df[cols] # changes order of columns
filename = 'WineGroundTruth/Whites_Flight1.csv' # change according to sheet from ground truth excel file
df.to_csv(filename, index = False) # output dataframe to csv file
