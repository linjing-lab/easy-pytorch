# Copyright (c) 2024 linjing-lab

import selfies
import pandas
df_sel = pandas.read_csv("molecule.smi", sep='\t') # get smiles strings
dataset = list(df_sel.iloc[:,0])
for i, s in enumerate(dataset):
    sf = selfies.encoder(s)
    dataset[i] = sf
    df_sel.iloc[i,0] = sf
alphabet = selfies.get_alphabet_from_selfies(dataset) # now selfies
alphabet.add("[nop]")  # [nop] is a special padding symbol
alphabet.add('.') # symbol different in selfies, be counted in len_selfies, discard in get_alphabet_from_selfies
alphabet = list(sorted(alphabet))
pad_to_len = max(selfies.len_selfies(s) for s in dataset)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
for s in dataset:
    label, one_hot = selfies.selfies_to_encoding(
    selfies=s,
    vocab_stoi=symbol_to_idx,
    pad_to_len=pad_to_len,
    enc_type="both"
    )
    print(s, label) # or append [label, one_not] into df_sel named descriptor_selfies
df_sel.to_csv('molecule.selfies', sep='\t', index=False, header=False)