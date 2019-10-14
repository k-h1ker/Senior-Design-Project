import numpy as np
import pandas as pd

"""
    Inputs:
        wine_id: Wine ID assigned
        color: White or Red Wine
        alc: Alcohol (v/v%)
        red_sug: Reducing Sugar (g/100mL)
        vol_acid: Volatile Acidity (g/100mL)
        titra_acid: Titratable Acidity (g/100mL)
        obrix_gal: OBrix (assuming this is the value from GAL Analysis)
        ph: pH
        va: VA (mg/L)
        obrix_must: OBrix (assuming this is the value from Initial Must Analysis)
        tot_sulfur_must: Total Sulfur (mg/L) (assuming this is the value from Initial Must Analysis)
        free_sulfur: Free Sulfur (mg/L)
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
        eth_but: Ethyl butanoate (ppb)
        eth_hex: Ethyl hexanoate (ppb)
        eth_isobut: Ethyl isobutyrate (ppb)
        eth_isoval: Ethyl isovalerate (ppb)
        eth_oct: Ethyl octanoate (ppb)
        iso_ace: Isoamyl acetate (ppb)
        lin_aroma: Linalool (ppb) (assuming this is the value from Wine Aroma Analysis)
        ibmp_aroma: IBMP (ppt) (assuming this is the value from Wine Aroma Analysis)
        caft_acid: Caftaric Acid (mg/L)
        catechin: Catechin (mg/L)
        grp: GRP (mg/L)

    Outputs:
        CSV file containing ground truth data for all wine samples

"""

    # 6 rows x 37 columns
    # values plugged in manually :( , excel sheet format not easy to automate data extraction
    # VALUES FOR ALL FLIGHTS OF WINE ARE IN backup.txt (not posted in GitHub for confidentiality)

wine_id           = ['M068',    'M037',   'M065',  'M072',    'M058',   'M055']
color             = ['red',     'red',    'red',   'red',     'red',    'red']
alc               = [12.53,     14.98,    15.55,   14.13,     14.56,    12.13]
red_sug           = ['<0.05',   '<0.05',  0.06,    0.05,      0.09,     '<0.05']
vol_acid          = [0.030,     0.038,    0.067,   0.029,     0.030,    0.081]
titra_acid        = [0.49,      0.61,     0.66,    0.62,      0.6,      0.53] # defaulting to Chemistry value , named twice in Reds
obrix_gal         = [22.2,      25.1,     26.9,    24.5,      26.7,     24.4]
ph                = [3.98,      4.14,     4.42,    3.60,      3.84,     4.31] # two named pH in Reds, defaulting to Must Analysis pH value
va                = ['<0.005',  0.005,    0.019,   '<0.005',  '<0.005', 0.014]
obrix_must        = [20.4,      26.2,     26.4,    24.9,      27.3,     27.3]
tot_sulfur_must   = [61,        175,      353,     106,       420,      '<5']
free_sulfur       = [23,        22,       36,      26,        23,       99]
copper            = ['<0.1',    0.2,      '<0.2',  0.24,      '<0.1',   0.21]
tot_sulfur_chem   = [55,        55,       74,      59,        55,       142]
mal_acid          = [525,       1042,     1877,    2133,      691,      428]
ibmp_gal          = [0,         0,        0,       0,         0,        0]
b_dama_gal        = [66,        56,       78,      27,        60,       30]
lin_gal           = [12,        30,       15,      5,         32,       6] # named 'bound linalool' in Reds
nerol             = [0,         12,       4,       0,         8,        0]
ta                = [0.25,      0.3,      0.4,     0.5,       0.38,     0.72]
nh3               = [39,        '<5',     71,      45,        79,       6]
nopa              = [166,       65,       433,     154,       222,      687]
yan               = [205,       70,       504,     199,       301,      693] # defaulting to value in the Must analysis, Red wines have two YAN values
hexanol1          = [1761,      1015,     1431,    1594,      1835,     1966]
methyl_ace2       = [134,       251,      118,     198,       185,      30]
phenyl_ace2       = [173,       690,      312,     379,       572,      91]
eth_but           = [255,       309,      228,     254,       536,      151]
eth_hex           = [309,       376,      318,     334,       487,      375]
eth_isobut        = [38,        99,       29,      55,        113,      22]
eth_isoval        = [6,         17,       5,       8,         15,       3]
eth_oct           = [396,       555,      364,     447,       646,      352]
iso_ace           = [1595,      2633,     1770,    2456,      2826,     486]
lin_aroma         = [4,         14,       6,       5,         16,       105]
ibmp_aroma        = ['<0.5',    '<0.5',   1.2,     '<0.5',    0.8,      '<0.5']
caft_acid         = [21,        29,       13,      20,        24,       54]
catechin          = [24,        12,       9,       15,        22,       112]
grp               = [4,         5,        6,       9,         7,        8]

properties = [alc, red_sug, vol_acid,
    titra_acid, obrix_gal, ph, va, obrix_must, tot_sulfur_must, free_sulfur,
    copper, tot_sulfur_chem, mal_acid, ibmp_gal, b_dama_gal, lin_gal, nerol,
    ta , nh3, nopa, yan, hexanol1, methyl_ace2, phenyl_ace2, eth_but, eth_hex,
    eth_isobut, eth_isoval, eth_oct, iso_ace, lin_aroma, ibmp_aroma, caft_acid, catechin, grp]

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
    'Volatile Acidity (g/100mL)': vol_acid,
    'Titratable Acidity (g/100mL)': titra_acid,
    'OBrix (GAL)': obrix_gal,
    'pH': ph,
    'VA (mg/L)': va,
    'OBrix (Must)': obrix_must,
    'Total Sulfur (mg/L) (Must)': tot_sulfur_must,
    'Free Sulfur (mg/L)': free_sulfur,
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
    'Ethyl butanoate (ppb)': eth_but,
    'Ethyl hexanoate (ppb)': eth_hex,
    'Ethyl isobutyrate (ppb)': eth_isobut,
    'Ethyl isovalerate (ppb)': eth_isoval,
    'Ethyl octanoate (ppb)': eth_oct,
    'Isoamyl acetate (ppb)': iso_ace,
    'Linalool (ppb) (Aroma)': lin_aroma,
    'IBMP (ppt) (Aroma)': ibmp_aroma,
    'Caftaric Acid (mg/L)': caft_acid,
    'Catechin (mg/L)': catechin,
    'GRP (mg/L)': grp
}

df = pd.DataFrame(data_dict) # python dictionary -> pandas dataframe
cols = ['Wine ID', 'Color', 'Alcohol (v/v%)', 'Reducing Sugar (g/100mL)',
        'Volatile Acidity (g/100mL)', 'Titratable Acidity (g/100mL)',
        'OBrix (GAL)', 'pH', 'VA (mg/L)', 'OBrix (Must)', 'Total Sulfur (mg/L) (Must)',
        'Free Sulfur (mg/L)', 'Copper (mg/L)', 'Total Sulfur (mg/L) (Chemistry)',
        'Malic Acid (mg/L)', 'IBMP (ppt) (GAL)', 'b-damascenone (ppb) (GAL)', 'Linalool (ppb) (GAL)',
        'Nerol (ppb)', 'TA (g/100mL)', 'NH3 (mg/L)', 'NOPA (ppm)', 'YAN (ppm)', '1-Hexanol (ppb)',
        '2-Methylbutyl acetat (ppb)', '2-Phenylethyl acetate (ppb)',
        'Ethyl butanoate (ppb)', 'Ethyl hexanoate (ppb)', 'Ethyl isobutyrate (ppb)', 'Ethyl isovalerate (ppb)',
        'Ethyl octanoate (ppb)', 'Isoamyl acetate (ppb)', 'Linalool (ppb) (Aroma)', 'IBMP (ppt) (Aroma)',
        'Caftaric Acid (mg/L)', 'Catechin (mg/L)', 'GRP (mg/L)']
df = df[cols] # changes order of columns
filename = 'WineGroundTruth/Reds_Flight7.csv' # change according to sheet from ground truth excel file
df.to_csv(filename, index = False) # output dataframe to csv file
