# Copyright (c) 2024 linjing-lab

import selfies
import pandas
df_sel = pandas.read_csv("molecule.smi", sep='\t', header=None) # get smiles strings
dataset = list(df_sel.iloc[:,0])
def gen_selfies():
    for i, s in enumerate(dataset):
        sf = selfies.encoder(s)
        dataset[i] = sf
        df_sel.iloc[i,0] = sf
    df_sel.to_csv('molecule.selfies', sep='\t', index=False, header=False)
gen_selfies()
# generate descriptor_selfies
alphabet = selfies.get_alphabet_from_selfies(dataset)
alphabet.add("[nop]")  # [nop] is a special padding symbol
alphabet.add('.') # symbol different in selfies, be counted in len_selfies, discard in get_alphabet_from_selfies
alphabet = list(sorted(alphabet))
pad_to_len = max(selfies.len_selfies(s) for s in dataset)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
labels, one_hots = [], []
for s in dataset:
    label, one_hot = selfies.selfies_to_encoding(
    selfies=s,
    vocab_stoi=symbol_to_idx,
    pad_to_len=pad_to_len,
    enc_type="both"
    )
    labels.append(label)
    one_hots.append(one_hot)
new_descrips = {'id': list(df_sel.iloc[:,1]), 'label': labels, 'one_hot': one_hots}
descriptor_selfies = pandas.DataFrame(new_descrips)
descriptor_selfies.to_csv('descriptor_selfies.csv', sep='\t', index=False)