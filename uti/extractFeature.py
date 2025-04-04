#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 23/02/2024
# Author: Sadettin Y. Ugurlu


import os

def extract_pocket_number(file_path):
    # Dosya adını al
    file_name = os.path.basename(file_path)
    # 'pocket' kelimesinden sonra gelen kısmı bul
    pocket_index = file_name.find('pocket')
    if pocket_index != -1:
        # 'pocket' kelimesinden sonraki kısmı al
        pocket_str = file_name[pocket_index + len('pocket'):]
        # Rakamı bul
        pocket_num = ''
        for char in pocket_str:
            if char.isdigit():
                pocket_num += char
            else:
                break
        # Eğer en az bir rakam bulunduysa, pocket numarasını döndür
        if pocket_num:
            return int(pocket_num)
    # Eğer pocket numarası bulunamazsa veya dosya adında 'pocket' kelimesi yoksa None döndür
    return None


def get_pocket_data(file_path, pocket_num):
    pocket_data = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    pocket_start = False
    pocket_end = False
    current_pocket = None

    for line in lines:
        line = line.strip()  # Satırın başındaki ve sonundaki boşlukları temizle

        if line.startswith(f'Pocket {pocket_num}'):
            pocket_start = True
            current_pocket = f'Pocket {pocket_num}'
            pocket_data[current_pocket] = {}
        elif pocket_start and line == '':
            pocket_end = True
        elif pocket_start and not pocket_end:
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                pocket_data[current_pocket][key] = value

    if not pocket_data:
        print(f"Pocket {pocket_num} not found in the file.")


    print(pocket_data.get(f'Pocket {pocket_num}', None))
    return pocket_data.get(f'Pocket {pocket_num}', None)

get_pocket_data("/home/yavuz/yavuz_proje/allosteric_feature_selected/data/training/1AO0_out/1AO0_info.txt",1)