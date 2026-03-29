

# chromosome lengths (GRCh38)
chromosome_length={'chr1':248956422, 
                   'chr2':242193529, 
                   'chr3':198295559, 
                   'chr4':190214555, 
                   'chr5':181538259, 
                   'chr6':170805979,
                   'chr7':159345973, 
                   'chr8':145138636, 
                   'chr9':138394717, 
                   'chr10':133797422, 
                   'chr11':135086622, 
                   'chr12':133275309, 
                   'chr13':114364328, 
                   'chr14':107043718, 
                   'chr15':101991189, 
                   'chr16':90338345, 
                   'chr17':83257441, 
                   'chr18':80373285,
                   'chr19':58617616, 
                   'chr20':64444167, 
                   'chr21':46709983, 
                   'chr22':50818468, 
                   'chrX':156040895, 
                   'chrY':57227415}


def vectorize_variant_count( variant_df, bin_size=1000000, seq_col="chr", pos_col="pos"):
    """
    Vectorize the variant count in bins of specified size across the genome.
    Assumes a GRCh38 reference genome. 
    The output is a dictionary where keys are bin labels (e.g., 'chr1_0', 'chr1_1', etc.) 
    and values are the count of variants in each bin.
    """

    bins = []  # (chrom, bin_start, bin_end, bin_label) 
    bin_labels=[]

    counter=0
    for chrom, length in chromosome_length.items():
        for i in range(1, length, bin_size): # last bin for each chromosome might not equal 1MB depending on the chr length
            bin_label = f'{chrom}_{counter}'
            bins.append((chrom, i, min(i + bin_size, length), bin_label))
            bin_labels.append(bin_label)
            counter+=1
        counter = 0

    vector_out = {}
    for chrom, start, end, label in bins: 
        svec = (variant_df[seq_col] == chrom) & (variant_df[pos_col] <= end) & (variant_df[pos_col] >= start)
        vector_out[label] = len(variant_df[svec])
    
    return vector_out