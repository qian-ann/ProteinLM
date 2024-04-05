# [0:20] Amino Acids (sparse encoding)
'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
# Unknown residues are stored as an all-zero vector
# [20:50] hmm profile
# [50] Seq mask (1 = seq, 0 = empty)
# [51] Disordered mask (0 = disordered, 1 = ordered)
# [52] Evaluation mask (For CB513 dataset, 1 = eval, 0 = ignore)
# [53] ASA (isolated)
# [54] ASA (complexed)
# [55] RSA (isolated)
# [56] RSA (complexed)
# [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
# [65:67] Phi+Psi
# [67] ASA_max