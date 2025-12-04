import random
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def hamming_distance(fp1, fp2):
    return np.sum(fp1 != fp2)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def check_mol(mol):
    if len(mol.GetAtoms()) < 2:
        st.error("Only compounds with more than 2 atoms are available for input.")
        return False
    else:
        return True

def show_search_results(search_df):
    col1result, col2result = st.columns([1, 1])
    pubchem = search_df['PubChem_CID'].iloc[0]
    cas = search_df['CAS'].iloc[0]
    if pubchem is not None:
        col1result.markdown(f'PubChem link: **https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem}**')
    if cas is not None:
        col2result.markdown(f'CAS link: **https://commonchemistry.cas.org/detail?cas_rn={cas}**')

    canonize_mol = search_df['SMILES_Solute'].iloc[0]
    col1result.markdown(f'**Molecule from MixtureSolDB**')
    col1result.image(draw_molecule(canonize_mol))
    col1result.code(canonize_mol, language="smiles")
    n_sources = search_df['Source'].nunique()
    n_solvents_pairs = search_df[['Solvent1', 'Solvent2']].drop_duplicates().shape[0]
    n_entries = search_df.shape[0]
    st.markdown(f'### {n_entries} entries from {n_sources} data sources')
    col2result.markdown(f"""
    # Overall stats:
    * **{n_entries}** number of entries
    * **{n_sources}** number of data sources
    * **{n_solvents_pairs}** number of binary solvent mixtures
    """)
    t_298 = search_df[(search_df['Temperature_K'] > 297.5) & (search_df['Temperature_K'] < 298.5)]
    if t_298.shape[0] > 0:
        t_298_max = t_298['Solubility(mole_fraction)'].max()
        solvent1, solvent2, fraction_solvent1, fraction_type, max_solubility = t_298[t_298['Solubility(mole_fraction)'] == t_298_max][['Solvent1', 'Solvent2', 'Fraction_Solvent1', 'Fraction_Type', 'Solubility(mole_fraction)']].iloc[0].values.tolist()
        col2result.markdown('* Top solvent pair by maximum observed solubility near 298 K:')
        col2result.metric("Pair", f'{solvent1} + {solvent2}')
        col2result.metric(f"{fraction_type} fraction of {solvent1}", f'{fraction_solvent1}')
        col2result.metric("Max solubility (mole fraction)", f'{max_solubility}')
    dois = list(search_df['Source'].unique())
    for num, doi in enumerate(dois):
        st.markdown(f'### {num+1}. https://doi.org/{doi}')
        col1result, col2result = st.columns([1, 1])
        df_comp = search_df[(search_df['Source'] == doi) & (search_df['SMILES_Solute'] == canonize_mol)]
        solvent_pairs = df_comp[['Solvent1', 'Solvent2']].drop_duplicates().reset_index(drop=True)
        for i, (solv1, solv2) in enumerate(zip(solvent_pairs['Solvent1'], solvent_pairs['Solvent2'])):
            df_comp_pair = df_comp[(df_comp['Solvent1'] == solv1) & (df_comp['Solvent2'] == solv2)]
            fig_pair = px.scatter_3d(df_comp_pair, x='Fraction_Solvent1', y='Temperature_K', z='Solubility(mole_fraction)',
                                color='Solubility(mole_fraction)',
                                color_continuous_scale='Viridis',
                                title=f'{solv1} + {solv2}')
            fig_pair.update_traces(marker=dict(size=5))
            fig_pair.update_layout(scene=dict(xaxis_title=f'Fraction of {solv1}', yaxis_title='T,K', zaxis_title='Solubility (mole fraction)'))
            if i % 2 == 0:
                col1result.plotly_chart(fig_pair, key=f'{canonize_mol}{i}')
            else:
                col2result.plotly_chart(fig_pair, key=f'{canonize_mol}{i}')
    st.dataframe(search_df)


calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='MixtureSolDB', layout="wide")

df = pd.read_csv('MixtureSolDB.csv')
df['PubChem_CID'] = df['PubChem_CID'].astype('Int64')
compound_names = sorted(df['Compound_Name'].unique().tolist())
fda_compound_names = sorted(df[df['FDA_Approved'] == 'Yes']['Compound_Name'].unique().tolist())

df_smiles = pd.DataFrame({'SMILES_Solute': list(df['SMILES_Solute'].unique())})
df_smiles['mol'] = df_smiles['SMILES_Solute'].apply(Chem.MolFromSmiles)
df_smiles['mol_ecfp'] = df_smiles['mol'].apply(lambda x: calc(x))

pairs = np.sort(df[['Solvent1', 'Solvent2']].values, axis=1)
pairs_df = pd.DataFrame(pairs, columns=['SolventA', 'SolventB'])
pair_counts = pairs_df.value_counts().reset_index(name='count')
solute_pairs = np.sort(df[['SMILES_Solute', 'Solvent1', 'Solvent2']].values, axis=1)
solute_pairs_df = pd.DataFrame(solute_pairs, columns=['SMILES_Solute', 'SolventA', 'SolventB'])
solute_pair_counts = solute_pairs_df.value_counts().reset_index(name='count')
top20 = pair_counts.head(20)
top20['pairs'] = top20['SolventA'] + " + " + top20['SolventB']

n_entries = df.shape[0]
n_smiles = df['SMILES_Solute'].nunique()
n_sources = df['Source'].nunique()
n_solvents_pairs = pair_counts.shape[0]
n_solvents_solute_pairs = solute_pair_counts.shape[0]
t_min = df['Temperature_K'].min()
t_max = df['Temperature_K'].max()
col1intro, col2intro, col3intro = st.columns([2, 1, 2])
col1intro.markdown(f"""
# MixtureSolDB

Download MixtureSolDB: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15094979.svg)](https://doi.org/10.5281/zenodo.15094979)
                   
""")

col2intro.markdown(f"""
# Overall stats:
* **{n_entries}** number of entries
* **{n_smiles}** unique solute molecules
* **{n_sources}** literature sources
* **{n_solvents_pairs}** unique binary solvent mixtures
* **{n_solvents_solute_pairs}** unique solute-binary solvent systems
* **{t_min}-{t_max}** temperature range
""")

col3intro.markdown(f"""
# Contributing to the dataset:
We encourage researchers to contribute to further development of the dataset either by performing literature screenings in the future or by standardized data contributions from the laboratories from all around the world. 

To supply the data in any format as well as any other suggestions/ideas regarding the MixtureSolDB project please contact: [lewa.krasnovs@gmail.com](mailto:lewa.krasnovs@gmail.com)
""")

tabs = st.tabs(["Explore", "Search by Compound Name", "Search by Molecular Structure", "Random Solubility🎲"])

with tabs[0]:
    col1fig, col2fig = st.columns([1, 1])
    fig_sol = px.histogram(df, x='Solubility(mole_fraction)', nbins=64, title='Mole fraction solubility distribution in the MixtureSolDB')
    fig_sol.update_layout(yaxis_title='Number of entries')
    fig_sol.update_layout(xaxis_title='Solubility(mole fraction)')
    col1fig.plotly_chart(fig_sol)

    fig_log_sol = px.histogram(df, x='LogS(mole_fraction)', nbins=64, title='LogS(mole_fraction) distribution in the MixtureSolDB')
    fig_log_sol.update_layout(yaxis_title='Number of entries')
    fig_log_sol.update_layout(xaxis_title='Log10 Solubility (mol/mol)')
    col2fig.plotly_chart(fig_log_sol)

    fig_solv = px.bar(top20, x='pairs', y='count', text='count', title="Most popular solute-binary solvent systems by number of entries")
    fig_solv.update_layout(yaxis_title='Number of entries')
    fig_solv.update_layout(xaxis_title='Solvent pairs')
    fig_solv.update_layout(xaxis_tickangle=90)
    st.plotly_chart(fig_solv, use_container_width=True)

with tabs[1]:
    fda = st.checkbox("Only FDA Approved molecules")

    if fda:
        selected = st.selectbox(label='Choose molecule', options=fda_compound_names, index=None, placeholder='Paracetamol')
    else:
        selected = st.selectbox(label='Choose molecule', options=compound_names, index=None, placeholder='Paracetamol')
    if selected:
        search_df = df[(df['Compound_Name'] == selected)]
        search_df.reset_index(drop=True, inplace=True)
        show_search_results(search_df)
        

with tabs[2]:

    st.markdown("""Draw your molecule to get SMILES and search in the database:""")

    smile_code = st_ketcher(height=400)
    st.markdown(f"""### Your SMILES:""")
    st.markdown(f"``{smile_code}``")
    st.markdown(f"""### Copy and paste this SMILES into the corresponding box below:""")

    smiles = st.text_input(
            "SMILES",
            placeholder='c1ccc2c(c1)[nH]cn2',
            key='smiles')

    if st.button("Search in the database"):
        if smiles:
            mol = Chem.MolFromSmiles(smiles.strip())
            if (mol is not None):
                if check_mol(mol):
                    canonize_mol = Chem.MolToSmiles(mol)
                    search_df = df[(df['SMILES_Solute'] == canonize_mol)]
                    search_df.reset_index(drop=True, inplace=True)
                    if search_df.shape[0] == 0:
                        st.markdown('This molecule was not found in MixtureSolDB. Enter another SMILES.')
                    else:
                        st.markdown(f'### This compound was found in MixtureSolDB:')
                        show_search_results(search_df)
            else:
                st.error("Incorrect SMILES entered")
        else:
            st.error("Please enter SMILES of the compound")

with tabs[3]:
    if st.button("Get solubility of random molecule🎲"):
        selected = random.choice(compound_names)
        search_df = df[(df['Compound_Name'] == selected)]
        search_df.reset_index(drop=True, inplace=True)
        show_search_results(search_df)
