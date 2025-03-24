#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/09/2024
# Author: Sadettin Y. Ugurlu

import time
import os
from uti import prepare_biopython
from uti import prepare_pybiomed
import pandas as pd
from uti import prepare_mathfeature
from uti import prepare_interaction_features
import glob


# -Training proteins:
training=['1ytmA_BS02_ATP', '3b6rB_BS02_ADP', '3hpiA_BS01_SUC', '1cetA_BS01_CLQ', '1swkA_BS01_BTN', '1jq3A_BS01_AAT', '1sz3A_BS01_GNP', '2f9wA_BS01_PAU', '1szjG_BS01_NAD', '3egvA_BS01_SAH', '1oxmA_BS01_TC4', '1r55A_BS04_097', '3cpaA_BS01_III', '2bawA_BS01_B7G', '3h39A_BS01_ATP', '3b4yA_BS02_FLC', '2dzbA_BS01_HH2', '3e3sA_BS04_I3C', '2dkcA_BS01_16G', '1g6cA_BS02_IFP', '2zhzB_BS02_ATP', '1sqfA_BS01_SAM', '2g25A_BS01_TDK', '2hxmA_BS01_302', '2chzA_BS01_093', '2g97A_BS01_DGB', '3a0tA_BS01_ADP', '3a5rA_BS01_HC4', '1uamA_BS02_SAH', '2zdqA_BS02_ATP', '2zgzA_BS02_GNP', '2jbtA_BS02_4HP', '1mu0A_BS01_PHK', '1xpyC_BS02_NLQ', '2i56A_BS01_RNS', '1q8jA_BS01_C2F', '3lzzA_BS01_GDP', '2z0xA_BS01_5CA', '1ogoX_BS01_UUU', '1bq3A_BS01_IHP', '3kdiA_BS01_A8S', '1e5qA_BS02_SHR', '3h2kA_BS01_BOG', '1ncoB_BS01_CHR', '1xm6A_BS02_5RM', '3gdlA_BS01_UP6', '3h72A_BS01_SIA', '1goqA_BS01_XYP', '1zt9A_BS02_TRP', '1fthA_BS01_A3P', '4stdA_BS01_BFS', '2dyaB_BS01_ADP', '3ll3A_BS02_DXP', '2b99A_BS03_RDL', '1xqpA_BS02_8HG', '1pnfA_BS01_UUU', '3fwrA_BS01_ADP', '966cA_BS06_RS2', '3ts1A_BS01_TYA', '1arcA_BS01_TCK', '3ej0A_BS01_11X', '3dsrA_BS01_ADP', '1bsvA_BS01_NDP', '1kwcB_BS01_BPY', '1xz8A_BS02_UUU', '2rjcA_BS02_MES', '3i6cA_BS02_GIA', '2irxA_BS02_GTP', '1xoqA_BS02_ROF', '1vcuA_BS01_DAN', '1qhiA_BS01_BPG', '1gkcA_BS05_NFH', '1zdfA_BS02_UPG', '3manA_BS01_UUU', '3c2fA_BS01_PRP', '2z09A_BS01_ACP', '3h18A_BS01_PMS', '1jjeA_BS04_BYS', '3exsA_BS01_5RP', '3jynA_BS01_NDP', '1r1hA_BS02_BIR', '1pfkA_BS01_FBP', '1of1A_BS01_SCT', '2dveA_BS01_BT5', '3btsB_BS02_NAD', '1h9uA_BS01_LG2', '2ntyC_BS01_GDP', '1hwwA_BS02_SWA', '1atlA_BS03_0QI', '2chtF_BS01_TSA', '1lbyA_BS01_F6P', '1p77A_BS01_ATR', '1pw1A_BS01_HEL', '1y6bA_BS01_AAX', '2gfxA_BS01_PMN', '2e0nB_BS01_SAH', '2zgmB_BS01_LAT', '3cgyA_BS01_RDC', '1ixnA_BS02_G3P', '2dptA_BS02_PUY', '1cenA_BS01_BGC', '13gsA_BS01_GSH', '1tkyA_BS01_A3S', '2ateA_BS01_NIA', '3iiqA_BS03_TRT', '2hk1A_BS01_FUD', '2hblA_BS03_AMP', '3gqkA_BS01_ATP', '1x55A_BS01_NSS', '1cqfB_BS02_UUU', '1i7lA_BS01_ATP', '1kaqA_BS01_DND', '1a4kH_BS01_FRA', '2wn7A_BS01_NAD', '1a2kC_BS01_GDP', '1yfrA_BS02_ATP', '3gpoA_BS01_APR', '2hzqA_BS01_STR', '3i0dA_BS01_UDP', '1um0A_BS01_FMN', '1bs1A_BS03_DAA', '1r091_BS01_JEN', '1svwA_BS02_GTP', '1of8B_BS02_PEP', '1uyyA_BS03_BGC', '1lloA_BS01_UUU', '1x7pA_BS01_SAM', '1mehA_BS02_IMP', '2ggaA_BS02_GPJ', '3b3fA_BS01_SAH', '1k3yA_BS01_GTX', '2j8yA_BS01_PNM', '1b8uA_BS02_NAD', '3kjgA_BS01_ADP', '1mkaA_BS02_DAC', '1hfcA_BS04_PLH', '3kbnA_BS03_GLO', '1tjwA_BS01_AS1', '1f8gA_BS01_NAD', '2ovdA_BS01_DAO', '1ryaA_BS02_GDP', '3efvA_BS01_NAD', '2c96A_BS01_ATP', '2ywcA_BS01_XMP', '2artA_BS02_UUU', '2gteA_BS01_VA', '1x2bA_BS01_STX', '1o26B_BS04_UMP', '1ogkB_BS01_DUD', '2uyqA_BS01_SAM', '1mbzA_BS03_IOT', '2hixA_BS01_ATP', '2cx8A_BS01_SAH', '2r68A_BS01_SUP', '1yqyA_BS02_915', '3i8xA_BS01_GDP', '1sthA_BS02_THP', '3bynA_BS01_RAF', '2fzsB_BS01_CMQ', '2wzmA_BS01_NA7', '5galB_BS01_UUU', '2q6vA_BS01_UDP', '1e3vA_BS01_DXC', '3b6aA_BS01_ZCT', '1ltzA_BS02_HBI', '2yw2A_BS01_ATP', '1jsvA_BS01_U55', '3in1A_BS01_ADP', '2royA_BS01_P28', '1sgcA_BS01_III', '7dfrA_BS01_FOL', '3ek5A_BS01_GTP', '3d1gA_BS01_322', '2dttB_BS02_H4B', '2gwhA_BS02_PCI', '3bazA_BS01_NAP', '3iesA_BS01_M24', '1txcA_BS02_2AN', '2qv7A_BS02_ADP', '3adpA_BS01_NAI', '1wxiA_BS02_AMP', '3erkA_BS01_SB4', '1y3iA_BS01_NAD', '1epbB_BS01_REA', '2za1A_BS01_OMP', '1ffqA_BS01_UUU', '1k54A_BS01_HOQ', '1v3sA_BS01_ATP', '3ftfA_BS03_SAH', '1k0nA_BS01_GSH', '1v48A_BS01_HA1', '2csnA_BS01_CKI', '1sbyA_BS01_NAD', '1f7bA_BS01_NAU', '1n07A_BS01_ADP', '1br6A_BS01_PT1', '1r15A_BS01_N', '2w1aC_BS01_TSA', '2jgvD_BS01_ADP', '1ch8A_BS04_IMP', '2vaqA_BS01_VAW', '1bzyA_BS01_IMU', '2ixlA_BS01_TRH', '1ojzA_BS01_NAD', '4dfrA_BS01_MTX', '1dcpA_BS01_HBI', '3kakA_BS01_3GC', '3g1xA_BS01_U', '1v0yA_BS01_HI5', '3gidA_BS01_S1A', '3dzlB_BS02_3OC', '1wopA_BS02_FFO', '2ioaA_BS04_ADP', '1foaA_BS02_UD1', '3a2sX_BS01_SUC', '2ihzA_BS02_CSF', '2qyqA_BS01_PTR', '1c3jA_BS01_UDP', '1n46A_BS01_PFA', '1pkkB_BS01_DCP', '1wnzA_BS01_2VA', '2rkmA_BS01_III', '1i7zA_BS01_COC', '2qwiA_BS02_NAG', '1gsaA_BS05_GSH', '1n2vA_BS02_BDI', '1s19A_BS01_MC9', '1mmbA_BS05_BAT', '1ig3A_BS01_VIB', '3du4A_BS02_KAP', '2i4nB_BS01_5CA', '1c5iA_BS01_UUU', '1navA_BS01_IH5', '3ergA_BS01_GTS', '1theA_BS01_0E6', '1jylA_BS02_CDC', '2e9zA_BS04_UTP', '2simA_BS01_DAN', '1x6uA_BS01_DO8', '1mncA_BS04_PLH', '1ri1A_BS01_SAH', '3idoA_BS01_EPE', '2hl0A_BS01_A3S', '3stdA_BS01_MQ0', '1jclB_BS01_HPD', '1xwqA_BS01_XYP', '1znzA_BS01_GDP', '1ddtA_BS01_APU', '2qttA_BS01_ADE', '2qzzA_BS02_EMF', '2fk8A_BS01_SAM', '1lspA_BS01_BUL', '2qehA_BS01_SRO', '2fhjA_BS03_H4Z', '1ucdA_BS01_U5P', '3hp8A_BS02_SUC', '3gh6A_BS01_GSH', '1thlA_BS06_0DB', '3cagA_BS01_ARG', '1e4gT_BS01_ATP', '3ku1A_BS01_SAM', '2ed4A_BS02_NAD', '1q41A_BS01_IXM', '3d4pA_BS01_NAD', '1c1hA_BS01_MMP', '5stdA_BS02_UNN', '1vh3A_BS01_CMK', '1rzuA_BS01_ADP', '2cwhB_BS02_NDP', '1jxmA_BS01_5GP', '1mxiA_BS01_SAH', '2r7aA_BS01_HEM', '4f10A_BS01_UUU', '1l7fA_BS03_BCZ', '1zajA_BS01_M2P', '1lzsA_BS02_NAG', '148lE_BS02_UUU', '1zu0A_BS01_CBS', '2rfhA_BS02_23N', '2x60A_BS01_GTP', '3duwA_BS01_SAH', '1og1A_BS01_TAD', '3ldkA_BS01_SUC', '1tvpB_BS01_CBI', '1kgzB_BS03_PRP', '1xtbA_BS01_S6P', '2fsgA_BS01_ATP', '1g97A_BS02_UD1', '1afkA_BS01_PAP', '3hvlA_BS01_SRL']

current_dir = os.path.dirname(os.path.abspath(__file__))

mathfeature_path=f"{current_dir}/bin/MathFeature/methods"


def prepare_all_features(protein_list,input_file_name,fpocket_filename):
    first_interaction=prepare_interaction_features.interaction_1_pre(protein_list=protein_list,
                                                                     input_file_name=input_file_name,
                                                                     current_dir=current_dir,
                                                                     delete_files=True)
    exit()
    ####################################################################################################################
    # prepare biopython related features
    biopython_data=prepare_biopython.biopython_descriptor(protein_list=protein_list,
                         fixed_pocket_path=f"{current_dir}/data/{input_file_name}",
                         )

    ####################################################################################################################
    # prepare pybiomed related features
    qso_data = prepare_pybiomed.get_pybiomed_features_QSOSW_QSOgrant(
        fixed_pocket_path=f"{current_dir}/data/{input_file_name}",
        protein_list=protein_list)
    cdt_data = prepare_pybiomed.get_cdt(fixed_pocket_path=f"{current_dir}/data/{input_file_name}",
                                        protein_list=protein_list)
    aacomposition_data = prepare_pybiomed.get_aacomposition(
        fixed_pocket_path=f"{current_dir}/data/{input_file_name}",
        protein_list=protein_list)
    conjoin_data = prepare_pybiomed.get_conjointtriad(fixed_pocket_path=f"{current_dir}/data/{input_file_name}",
                                                      protein_list=protein_list)
    ####################################################################################################################
    os.chdir(current_dir)
    math_feature_fast = f"{input_file_name}.fasta"
    if os.path.exists(math_feature_fast):
        os.remove(math_feature_fast)
    _math_feature_csv = f"{input_file_name}_math_feature.csv"
    if os.path.exists(_math_feature_csv):
        os.remove(_math_feature_csv)


    prepare_mathfeature.math_feature(protein_list=protein_list,
             input_path=f"{current_dir}/data/{input_file_name}",
             output_filename=input_file_name)
    time.sleep(10)
    os.chdir(current_dir)

    os.system(
        f"python3 {mathfeature_path}/Kgap.py -i {input_file_name}.fasta -l protein -k 1 -bef 1 -aft 1 -seq 3 -o {input_file_name}_math_feature.csv")
    math_feature_data =pd.read_csv(f"{input_file_name}_math_feature.csv")
    math_feature_data.drop(columns=["label"], inplace=True) # label here is PROTEIN (in up os.system code: -l protein)
    math_feature_data.rename(columns={"nameseq": "protein_pocket"}, inplace=True)

    ####################################################################################################################
    # Load dataframes"""
    os.chdir(current_dir)
    fpocket_data = pd.read_csv(fpocket_filename)

    # Merge the dataframes sequentially
    frames = [first_interaction,
              biopython_data,
              qso_data,
              cdt_data,
              aacomposition_data,
              conjoin_data,
              math_feature_data,
              ]

    # Merge dataframes sequentially, accumulating the result
    merged_df = fpocket_data  # Start with the first dataframe
    for df in frames:
        merged_df = pd.merge(merged_df, df, on="protein_pocket", how="outer")


    # Print the first few rows of merged dataframe to check if data is present
    print("First few rows of merged dataframe:")
    print(merged_df.head())
    print(len(merged_df))
    # Save merged dataframe to CSV
    merged_df.to_csv(f"{input_file_name}_ready.csv", index=False)

prepare_all_features(protein_list=training,
                     input_file_name="training_290",
                     fpocket_filename="training.csv")
print("finished")
