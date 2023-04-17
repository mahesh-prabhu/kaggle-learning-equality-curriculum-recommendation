import pandas as pd
import dask.dataframe as dd
import datetime
import gc
import torch
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import GroupKFold
from functools import partialmethod

def progress_bar_control(show_progress_bars = True):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=(not show_progress_bars))

def pt(s):
    print(str(datetime.datetime.now()) + " : " + s)

def split_array(arr, num_splits):
    return [list(map(tuple, a)) for a in np.array_split(np.array(arr), num_splits)]

def flatten_l(l):
    return [item for sublist in l for item in sublist]

def list_diff_with_ordering_maintained(l1, l2):
    """
    Diff between l1 and l2 with the ordering in l1 maintained.
    """
    a1 = np.array(l1)
    a2 = np.array(l2)

    # Below from https://stackoverflow.com/questions/46261671/use-numpy-setdiff1d-keeping-the-order
    return list(a1[~np.in1d(a1,a2)])

# Below ceiling division function from https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b):
    return -(a // -b)

def sum_pair_list(pairs_list):
    total_sum = 0
    for i, c in pairs_list:
        total_sum += c
    return total_sum

# Below code from https://stackoverflow.com/questions/35517051/split-a-list-of-numbers-into-n-chunks-such-that-the-chunks-have-close-to-equal
def split_pair_list(lst, chunks):
    #print(lst)
    chunks_yielded = 0
    #total_sum = sum(lst)
    total_sum = sum_pair_list(lst)
    avg_sum = total_sum/float(chunks)
    chunk = []
    chunksum = 0
    sum_of_seen = 0

    for i, item in enumerate(lst):
        #print('start of loop! chunk: {}, index: {}, item: {}, chunksum: {}'.format(chunk, i, item, chunksum))
        if chunks - chunks_yielded == 1:
            #print('must yield the rest of the list! chunks_yielded: {}'.format(chunks_yielded))
            yield chunk + lst[i:]
            # raise StopIteration
            # MP: why raise exception? Just break from the loop
            break

        to_yield = chunks - chunks_yielded
        chunks_left = len(lst) - i
        if to_yield > chunks_left:
            #print('must yield remaining list in single item chunks! to_yield: {}, chunks_left: {}'.format(to_yield, chunks_left))
            if chunk:
                yield chunk
            yield from ([x] for x in lst[i:])
            # raise StopIteration
            # MP: why raise exception? Just break from the loop
            break

        sum_of_seen += item[1]
        if chunksum < avg_sum:
            #print('appending {} to chunk {}'.format(item, chunk))
            chunk.append(item)
            chunksum += item[1]
        else:
            #print('yielding chunk {}'.format(chunk))
            yield chunk
            # update average expected sum, because the last yielded chunk was probably not perfect:
            avg_sum = (total_sum - sum_of_seen)/(to_yield - 1)
            chunks_yielded += 1
            chunksum = item[1]
            chunk = [item]


def load_data_as_folds(num_folds, dirname_folds, filename_prefix_folds, contents_df, topics_df):
    # Load all the folds
    folds_dfs = []
    pt('Loading folds with ids ...')
    for i in range(0, num_folds):
        df = pd.read_csv(f'{dirname_folds}/{filename_prefix_folds}{i}.csv')
        folds_dfs.append(df)

    # Load the content and topic data
    # pt('Loading content and topic data ...')
    # contents_df = pd.read_feather(filename_contents_feather)
    # topics_df   = pd.read_feather(filename_topics_feather)

    pt('Merge content and topic data for each fold ...')
    # Merge the data
    folds_with_data_dfs = []
    for i in range(0, num_folds):
        pt(f'Merging fold {i} ...')
        df = pd.merge(folds_dfs[i], topics_df  , how='inner', on='t_id')
        df = pd.merge(df          , contents_df, how='inner', on='c_id')
        folds_with_data_dfs.append(df)
        
    return folds_with_data_dfs

def get_triplet_data_for_triplet_fold(fold_df, contents_df, topics_df, drop_ids = True, shuffle = False):
    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd
    #print(dm)
    df = dm.merge(fold_df, topics_df  , how='inner', on='t_id')
    df = dm.merge(df     , contents_df, how='inner', left_on='pos_c_id', right_on='c_id')
    df = dm.merge(df     , contents_df, how='inner', left_on='neg_c_id', right_on='c_id', suffixes = ('_p', '_n'))

    if (drop_ids):
        id_cols = [col for col in df.columns if '_id' in col]
        df = df.drop(id_cols, axis = 1)
        #df.drop('c_id_p', axis = 1, inplace=True)
        #df.drop('c_id_n', axis = 1, inplace=True)

    if shuffle:
        df = df.sample(frac = 1) # not sure if this will work with dask
        
    return df

def get_triplet_data_from_triplet_folds(folds, dirname_folds, filename_prefix_folds, contents_df, topics_df, drop_ids = True, stack = False, shuffle = False):

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd
    
    # Load all the folds
    folds_dfs = []
    pt(f'Getting triplet data from folds : {folds}')
    for i in folds:
        df = dm.read_csv(f'{dirname_folds}/{filename_prefix_folds}{i}.csv')
        folds_dfs.append(df)

    pt('Merge content and topic data for each fold to get triplet data...')
    # Merge the data
    train_data_df = None
    for i, f in enumerate(folds):
        pt(f'Getting triplet data for fold {f} ...')
        df = get_triplet_data_for_triplet_fold(folds_dfs[i], contents_df, topics_df, drop_ids, shuffle)
        if (train_data_df is None):
            if (stack):
                train_data_df = df
            else:
                train_data_df = [df]
        else:
            if (stack):
                m1 = train_data_df.memory_usage(deep=True).sum()
                m2 = df.memory_usage(deep=True).sum()
                pt(f'Stacking triplet data ... memory usage of two dataframes : {m1} , {m2}')
                train_data_df = dm.concat([train_data_df, df], axis = 0)
                del df
                gc.collect()
            else:
                train_data_df.append(df)
    return train_data_df

def get_bin_data_for_triplet_fold(triplet_fold_df, contents_df, topics_df, drop_ids = True, shuffle = False, drop_negatives = False):

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd

    # Extract neg and pos content from fold_df to create a new fold_df
    pos_fold_df = triplet_fold_df[['t_id','pos_c_id']].copy()
    pos_fold_df = pos_fold_df.drop_duplicates()
    # rename pos_c_id to c_id
    pos_fold_df = pos_fold_df.rename(columns={'pos_c_id':'c_id'})
    # add label 1 columns
    pos_fold_df['label'] = 1

    neg_fold_df = triplet_fold_df[['t_id','neg_c_id']].dropna().copy()
    # rename neg_c_id to c_id
    neg_fold_df = neg_fold_df.rename(columns={'neg_c_id':'c_id'})
    # add label 0 columns
    neg_fold_df['label'] = 0

    # stack both dfs
    if drop_negatives:
        fold_df = pos_fold_df
    else:
        fold_df = dm.concat([pos_fold_df, neg_fold_df],axis=0)

    # change the label column to uint8
    fold_df = fold_df.astype({'label':'uint8'})
    
    df = dm.merge(fold_df , topics_df  , how='inner', on='t_id')
    df = dm.merge(df      , contents_df, how='inner', on='c_id')

    if (drop_ids):
        id_cols = [col for col in df.columns if '_id' in col]
        df = df.drop(id_cols, axis = 1)

    if shuffle:
        df = df.sample(frac = 1) # not sure if this will work with dask
        
    return df

def convert_df_to_torch(df, t_cols, c_cols):
    pt('Converting DF to torch tensor ...')

    torch_topic = torch.tensor(df[t_cols].values)
    df.drop(t_cols, axis = 1, inplace = True)

    torch_content = torch.tensor(df[c_cols].values)
    df.drop(c_cols, axis = 1, inplace = True)

    label_col = ['label']

    torch_label = torch.tensor(df[label_col].values)
    df.drop(label_col, axis = 1, inplace = True)

    return torch_topic, torch_content, torch_label
    

def get_bin_data_from_triplet_folds(folds, dirname_folds, filename_prefix_folds, contents_df, topics_df, drop_ids = True, stack = False, shuffle = False, to_torch = False, drop_negatives = False, only_emb_cols = False):

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd
    
    # Load all the folds
    folds_dfs = []
    pt(f'Getting bin data from folds : {folds}')
    for i in folds:
        df = dm.read_csv(f'{dirname_folds}/{filename_prefix_folds}{i}.csv')
        folds_dfs.append(df)

    # Below defines ordering of columns
    if not only_emb_cols:
        t_cols = [col for col in topics_df.columns if not '_id' in col]
    else:
        t_cols = [col for col in topics_df.columns if '_emb_' in col]
    pt(f'First 10 topic cols: {t_cols[0:10]}')
    if not only_emb_cols:
        c_cols = [col for col in contents_df.columns if not '_id' in col]
    else:
        c_cols = [col for col in contents_df.columns if '_emb_' in col]
    pt(f'First 10 content cols: {c_cols[0:10]}')

    pt('Merge content and topic data for each fold to get bin data...')
    # Merge the data
    val_data_df = None
    for i, f in enumerate(folds):
        pt(f'Getting bin data for fold {f} ...')
        df = get_bin_data_for_triplet_fold(folds_dfs[i], contents_df, topics_df, drop_ids, drop_negatives)
        if to_torch:
            df = convert_df_to_torch(df, t_cols, c_cols)
        if (val_data_df is None):
            if (stack):
                val_data_df = df
            else:
                val_data_df = [df]
        else:
            if (stack):
                if to_torch:
                    pt(f'Concatenating torch tensors ... ')
                    val_data_df = (torch.cat([val_data_df[0], df[0]],0), torch.cat([val_data_df[1], df[1]],0), torch.cat([val_data_df[2], df[2]],0))
                else:
                    m1 = val_data_df.memory_usage(deep=True).sum()
                    m2 = df.memory_usage(deep=True).sum()
                    pt(f'Stacking bin data ... memory usage of two dataframes : {m1} , {m2}')
                    val_data_df = dm.concat([val_data_df, df], axis = 0)
                del df
                gc.collect()
            else:
                val_data_df.append(df)
        
    return val_data_df


def get_unique_batch_size_partitioning(pairs, batch_size):
    def insert_into_bin(t_id, c_id, b_id, all_bins, c_t_id_to_bin_ids):
        # Add pair to the bin
        s = all_bins[b_id]
        s.append((t_id, c_id))

        # Add the bin-id to list of bins that t_id belongs to
        if(t_id not in c_t_id_to_bin_ids):
            c_t_id_to_bin_ids[t_id] = []
        c_t_id_to_bin_ids[t_id].append(b_id)

        # Add the bin-id to list of bins that c_id belongs to
        if(c_id not in c_t_id_to_bin_ids):
            c_t_id_to_bin_ids[c_id] = []
        c_t_id_to_bin_ids[c_id].append(b_id)

    
    active_bins = []
    all_bins = {} # bin-id to bin(list) mapping
    c_t_id_to_bin_ids = {}
    for (t_id, c_id) in pairs:
        #  get t_id_bins
        #  get c_id_bins
        t_id_bins = c_t_id_to_bin_ids[t_id] if t_id in c_t_id_to_bin_ids else []
        c_id_bins = c_t_id_to_bin_ids[c_id] if c_id in c_t_id_to_bin_ids else []

        # no_t_id_c_id_bins = active_bins - t_id_bins - c_id_bins
        no_t_id_c_id_bins = list_diff_with_ordering_maintained(active_bins, t_id_bins)
        no_t_id_c_id_bins = list_diff_with_ordering_maintained(no_t_id_c_id_bins,c_id_bins)

        #  Pick a bin 'b' from no_t_id_c_id_bins, if no_t_id_c_id_bins is empty, create a new bin 'b'. Add 'b' to active bin list, insert (t_id, c_id) in bin 'b'.
        if len(no_t_id_c_id_bins) == 0:
            # Create a new bin
            b_id = len(all_bins)
            all_bins[b_id] = []
            active_bins.append(b_id)
        else:
            # b_id = random.sample(no_t_id_c_id_bins, 1)[0]
            # Randomly picking these bins, gave us a lot of bins less
            # than the batch size.
            # Instead  picking the earliest empty bin resulted in only the
            # last bin being less than the batch size.
            b_id = no_t_id_c_id_bins[0]

        #  Add bin 'b' to t_id's and c_id's bin list
        insert_into_bin(t_id, c_id, b_id, all_bins, c_t_id_to_bin_ids)
        
        #  If bin 'b' has batch_size elements remove it from list of active bins
        if (len(all_bins[b_id]) == batch_size):
            active_bins.remove(b_id)

    partitioned_pairs = []

    for _, b  in all_bins.items():
        partitioned_pairs.append(b)

    return partitioned_pairs


def strip_neg_data_from_triplet_fold(triplet_fold_df):

    # Extract neg and pos content from fold_df to create a new fold_df
    pos_fold_df = triplet_fold_df[['t_id','pos_c_id']].copy()
    pos_fold_df = pos_fold_df.drop_duplicates()
    # rename pos_c_id to c_id
    pos_fold_df = pos_fold_df.rename(columns={'pos_c_id':'c_id'})

    return pos_fold_df

def convert_df_to_torch_no_labels(df, t_cols, c_cols):

    torch_topic = torch.tensor(df[t_cols].values)
    df.drop(t_cols, axis = 1, inplace = True)

    torch_content = torch.tensor(df[c_cols].values)
    df.drop(c_cols, axis = 1, inplace = True)

    return torch_topic, torch_content

def validate_partitioning(partitioned_pairs, batch_size):
    pt('Validating the partitioning ... ')
    ids_set = [None]*len(partitioned_pairs) 
    for i, pair_list in enumerate(partitioned_pairs):
        if (len(pair_list) < batch_size):
            print(f'Pair list #{i} has {len(pair_list)} items which is less than the batch-size({batch_size})')
        ids_set[i] = set([])
        for p in pair_list:
            if (p[0] in ids_set[i]):
                pt(f'Item reappears in partition {i}')
            else:
                ids_set[i].add(p[0])
            if (p[1] in ids_set[i]):
                pt(f'Item reappears in partition {i}')
            else:
                ids_set[i].add(p[1])

    # pt('Checking if the partitions are mutually exclusive ... ')
    # for i in range(0, len(ids_set)-1):
    #     for j in range(i+1, len(ids_set)):
    #         if (len(ids_set[i].intersection(ids_set[i])) != 0):
    #             pt(f'Partitions {i} and {j} are not mutually exclusive.')
    
def get_partitioned_data_from_triplet_folds(folds, dirname_folds, filename_prefix_folds, contents_df, topics_df, batch_size, only_emb_cols = False):
    """
    This function takes the triplet fold data and returns the merged data only for positive pairs. The positive (t_id, c_id) pairs are partitined into batch_size boundaries, such that the t_id and c_id from a pair appears only once within the batch/partition. This kind of partitioning is useful for multiple negatives ranking loss (See assumption of the pairs list for multiple negatives ranking loss).
    """

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd
    
    # Load all the folds
    folds_dfs = []
    pt(f'Getting bin data from folds : {folds}')
    for i in folds:
        df = dm.read_csv(f'{dirname_folds}/{filename_prefix_folds}{i}.csv')
        folds_dfs.append(df)

    
    pt('Stripping neg data from folds ...')
    pos_folds_df = None
    for i, f in enumerate(folds):
        df = strip_neg_data_from_triplet_fold(folds_dfs[i])

        if (pos_folds_df is None):
            pos_folds_df = df
        else:
            pt('Stacking folds ... ')
            pos_folds_df = dm.concat([pos_folds_df, df], axis = 0)

    # Randomly shuffle the folds
    #pos_folds_df = pos_folds_df.sample(frac = 1)

    # Convert the df into a list of pairs
    pos_pairs = list(pos_folds_df.itertuples(index=False, name=None))
    random.shuffle(pos_pairs)

    pt(f'Partitioning {len(pos_pairs)} positive pairs ... ')
    partitioned_pairs = get_unique_batch_size_partitioning(pos_pairs, batch_size)
    
    pt(f'Total number of partitions : {len(partitioned_pairs)}')

    validate_partitioning(partitioned_pairs, batch_size)

    # Merge in the topic and content data
    # take one partition at a time and create the pytorch data

    # Below defines ordering of columns
    if not only_emb_cols:
        t_cols = [col for col in topics_df.columns if not '_id' in col]
    else:
        t_cols = [col for col in topics_df.columns if '_emb_' in col]
    pt(f'First 10 topic cols: {t_cols[0:10]}')
    if not only_emb_cols:
        c_cols = [col for col in contents_df.columns if not '_id' in col]
    else:
        c_cols = [col for col in contents_df.columns if '_emb_' in col]
    pt(f'First 10 content cols: {c_cols[0:10]}')


    pt('Merging partitioned_pairs into a df ... ')
    corr_df = None
    for pairs_list in tqdm(partitioned_pairs):
        if (corr_df is None):
            corr_df = dm.DataFrame(pairs_list, columns = ['t_id','c_id'])
        else:
            corr_df = dm.concat([corr_df, dm.DataFrame(pairs_list, columns = ['t_id','c_id'])], axis = 0)

    pt('Merging in topic and content data into the folds data ... ')
    df = dm.merge(corr_df , topics_df  , how='inner', on='t_id')
    df = dm.merge(df      , contents_df, how='inner', on='c_id')

    # We'll drop the id columns
    id_cols = [col for col in df.columns if '_id' in col]
    df = df.drop(id_cols, axis = 1)

    pt('Converting data to torch ... ')
    data_torch = convert_df_to_torch_no_labels(df, t_cols, c_cols)

    return data_torch


def get_pos_corr_subsets_for_triplet_fold(triplet_fold_df, contents_df, topics_df):

    # Extract pos content from fold_df to create a new fold_df
    fold_df = triplet_fold_df[['t_id','pos_c_id']].copy()
    fold_df = fold_df.drop_duplicates()
    # rename pos_c_id to c_id
    fold_df = fold_df.rename(columns={'pos_c_id':'c_id'})
    
    topics_subset_df   = topics_df[topics_df['t_id'].isin(set(fold_df['t_id'].to_list()))].copy().reset_index(drop=True).reset_index()

    contents_subset_df = contents_df[contents_df['c_id'].isin(set(fold_df['c_id'].to_list()))].copy().reset_index(drop=True).reset_index()

    idx_corr_subset_df = pd.merge(fold_df, topics_subset_df, on = 't_id', how = 'inner')[['index', 'c_id']].copy().rename(columns={'index':'t_idx'})
    idx_corr_subset_df = pd.merge(idx_corr_subset_df, contents_subset_df, on = 'c_id', how = 'inner')[['t_idx', 'index']].copy().rename(columns={'index':'c_idx'})

    corr_subset_df = idx_corr_subset_df.groupby('t_idx')['c_idx'].apply(list)
        
    return topics_subset_df, contents_subset_df, corr_subset_df

def convert_subset_dfs_to_tensor(topics_subset_df, contents_subset_df, t_cols, c_cols):
    pt('Converting DF to torch tensor ...')

    topics_ten = torch.tensor(topics_subset_df[t_cols].values)

    contents_ten = torch.tensor(contents_subset_df[c_cols].values)

    return topics_ten, contents_ten
    

def get_pos_corr_data_from_triplet_folds(folds, dirname_folds, filename_prefix_folds, contents_df, topics_df, only_emb_cols = False):
    
    # Load all the folds
    folds_dfs = []
    pt(f'Getting bin data from folds : {folds}')
    for i in folds:
        df = pd.read_csv(f'{dirname_folds}/{filename_prefix_folds}{i}.csv')
        folds_dfs.append(df)

    # Below defines ordering of columns
    if not only_emb_cols:
        t_cols = [col for col in topics_df.columns if not '_id' in col]
    else:
        t_cols = [col for col in topics_df.columns if '_emb_' in col]
    pt(f'First 10 topic cols: {t_cols[0:10]}')
    if not only_emb_cols:
        c_cols = [col for col in contents_df.columns if not '_id' in col]
    else:
        c_cols = [col for col in contents_df.columns if '_emb_' in col]
    pt(f'First 10 content cols: {c_cols[0:10]}')

    # Merge the data
    val_data = None
    for i, f in enumerate(folds):
        pt(f'Getting pos corr data for fold {f} ...')
        topics_subset_df, contents_subset_df, corr_subset_df = get_pos_corr_subsets_for_triplet_fold(folds_dfs[i], contents_df, topics_df)
        topics_ten, contents_ten = convert_subset_dfs_to_tensor(topics_subset_df, contents_subset_df, t_cols, c_cols)
        if val_data is None:
            val_data = (topics_ten, contents_ten, corr_subset_df)
        else:
            pt(f'Concatenating torch tensors and df ... ')
            val_data = (torch.cat([val_data[0], topics_ten],0), torch.cat([val_data[1], contents_ten],0), pd.concat([val_data[2], corr_subset_df], axis = 0))
            
        
    return val_data


def triplet_fold_to_bin_fold_df(triplet_fold_df, drop_negatives = False):
    # Extract neg and pos content from fold_df to create a new fold_df
    pos_fold_df = triplet_fold_df[['t_id','pos_c_id']].copy()
    pos_fold_df = pos_fold_df.drop_duplicates()
    # rename pos_c_id to c_id
    pos_fold_df = pos_fold_df.rename(columns={'pos_c_id':'c_id'})
    # add label 1 columns
    pos_fold_df['label'] = 1

    neg_fold_df = triplet_fold_df[['t_id','neg_c_id']].dropna().copy()
    # rename neg_c_id to c_id
    neg_fold_df = neg_fold_df.rename(columns={'neg_c_id':'c_id'})
    # add label 0 columns
    neg_fold_df['label'] = 0

    # stack both dfs
    if drop_negatives:
        fold_df = pos_fold_df
    else:
        fold_df = pd.concat([pos_fold_df, neg_fold_df],axis=0)

    # change the label column to uint8
    fold_df = fold_df.astype({'label':'uint8'})

    return fold_df.reset_index(drop = True)


def get_all_parents(topics_parent_dict, t_id):
    parents = []
    while (t_id in topics_parent_dict):
        parent_t_id = topics_parent_dict[t_id]
        if (parent_t_id is not None and parent_t_id is not np.nan): # Roots will have None as parent
            parents.append(parent_t_id)
        else:
            break
        t_id = parent_t_id
    return parents
        
def get_all_parents_title_string(topics_parent_dict, topics_title_dict, t_id, sep_token):
    parents = get_all_parents(topics_parent_dict, t_id)
    parent_titles = list(map(lambda t_id: topics_title_dict[t_id], parents))
    all_parents_title_string = sep_token.join(parent_titles)
    return all_parents_title_string


def create_train_val_split_pos_corr(topics_df, content_df, correlations_df, num_folds, min_train_perc, use_topic_trees = False, random_seed = 0):
    G = nx.Graph()

    # Add t_ids to the graph
    # t_ids = topics_df['id'].to_list()
    # G.add_nodes_from(t_ids)

    if (use_topic_trees):
        pt(f'Add parent-child t_id relations as edges to the graph')
        edges = topics_df.dropna(subset=['parent'])[['id', 'parent']].apply(tuple, axis=1).tolist()
        G.add_edges_from(edges)
        pt(f'Total Topic Trees:')
        total_topic_trees = len(list(nx.connected_components(G)))
        print(total_topic_trees)
        

    pt(f'Creating topic->contents and contents->topics mappings.')
    correlations_df_dict = correlations_df.to_dict('index')
    correlations_dict = {} # maps topic_id -> content_ids list
    for k, v in correlations_df_dict.items():
        t_id = v['topic_id']
        c_ids = v['content_ids'].split()
        correlations_dict[t_id] = c_ids

    pt(f' Adding t_id, c_id relations as edges to the graph')
    for t_id, c_id_list in correlations_dict.items():
        G.add_edges_from([(t_id, c_id) for c_id in c_id_list])


    pt(f'Identify all the connected components')
    components = list(nx.connected_components(G))
    print(f'Num connected components: {len(components)}')

    
    pt(f'Creating id for each component')
    t_c_id_to_comp_id = {}
    for i, comp_nodes in enumerate(components):    
        for t_c_id in comp_nodes:
            t_c_id_to_comp_id[t_c_id] = i

    pt(f'Counting number of mappings in each component')
    comp_id_num_mappings = {}
    for t_id, c_id_list in correlations_dict.items():
        comp_id = t_c_id_to_comp_id[t_id]
        if comp_id in comp_id_num_mappings:
            comp_id_num_mappings[comp_id] += len(c_id_list)
        else:
            comp_id_num_mappings[comp_id] = len(c_id_list)

    # print(f'Comp-id to num pairs: {comp_id_num_mappings}')

    pt(f'Splitting components into {num_folds} folds')
    lst = list(comp_id_num_mappings.items())
    random.seed(random_seed)
    random.shuffle(lst)
    folds = []
    print('Expected avg fold size: {}'.format(sum_pair_list(lst)/float(num_folds)))
    fold_sizes = []
    for fold in split_pair_list(lst, num_folds):
        fold_sizes.append(sum_pair_list(fold))
        folds.append(fold)

    print(f'fold sizes: {fold_sizes}')
    total = sum([sum_pair_list(fold) for fold in folds])
    fold_percs = [(sum_pair_list(fold)/total)*100 for fold in folds]

    print(f'Fold percentage splits: {fold_percs}')

    
    fold_id_to_mappings = {}
    comp_id_to_mappings = {}
    for i, fold in enumerate(folds):
        mappings_list = []
        fold_id_to_mappings[i] = mappings_list
        for comp_id, count in fold:
            # Components that belong to the same fold point to the same mappings_list
            comp_id_to_mappings[comp_id] = mappings_list

    pt(f'Creating mapping lists for each fold.')
    for t_id, c_id_list in correlations_dict.items():
        mappings = [(t_id, c_id) for c_id in c_id_list]
        comp_id = t_c_id_to_comp_id[t_id]
        comp_id_to_mappings[comp_id] += mappings

    pt(f'Creating a dataframe with mappings for each fold.')
    fold_dfs = []
    for fold_id, mappings_list in fold_id_to_mappings.items():
        df = pd.DataFrame(mappings_list, columns=['t_id','c_id'])
        fold_dfs.append(df)

    fold_dfs = sorted(fold_dfs, key = lambda df: len(df), reverse = True)

    train_df = []
    val_df = []

    total = sum([len(df) for df in fold_dfs])
    fold_percs = [(len(df)/total)*100 for df in fold_dfs]
    cur_train_df_size = 0
    for i, fold_df in enumerate(fold_dfs):
        if (cur_train_df_size>= min_train_perc):
            val_df.append(fold_df)
        else:
            train_df.append(fold_df)
            cur_train_df_size += fold_percs[i]

    pt(f'***** Train DFs sizes *****')
    for df in train_df:
        size = len(df)
        perc = (size/total)*100
        print(f'{size} : {perc:.2f} ')

    pt(f'***** Val DFs sizes *****')
    for df in val_df:
        size = len(df)
        perc = (size/total)*100
        print(f'{size} : {perc:.2f} ')

    train_df = pd.concat(train_df, ignore_index = True, axis = 0)
    val_df = pd.concat(val_df, ignore_index = True, axis = 0)

    return train_df, val_df


def get_partitioned_data_for_pos_corr(pos_corr_df, topics_df, contents_df, batch_size, add_labels = False, do_partitioning = False, sort = False):
    """
    This function takes the poss corr df and returns the merged data. The positive (t_id, c_id) pairs are partitined into batch_size boundaries, such that the t_id and c_id from a pair appears only once within the batch/partition. This kind of partitioning is useful for multiple negatives ranking loss (See assumption of the pairs list for multiple negatives ranking loss).
    """

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd
    

    # Convert the df into a list of pairs

    if (do_partitioning):
        if (not sort):
            pt(f'Randomizing positive pairs ... ')
            pos_pairs = list(pos_corr_df.itertuples(index=False, name=None))
            random.shuffle(pos_pairs)
        else:
            pt(f'Sorting positive pairs based on highest correlations ... ')
            corr_df_pairs_list = list(pos_corr_df.groupby('t_id')['c_id'].apply(list).items()) # This gives a list of pairs of (t_id, [list of c_id])
            corr_df_pairs_list = sorted(corr_df_pairs_list, key = lambda pair: len(pair[1]), reverse = True)
            pos_pairs = [(t_id, c_id) for (t_id, c_ids) in corr_df_pairs_list for c_id in c_ids ]

        pt(f'Partitioning {len(pos_pairs)} positive pairs in partition of size: {batch_size}... ')
        partitioned_pairs = get_unique_batch_size_partitioning(pos_pairs, batch_size)
    
        pt(f'Total number of partitions : {len(partitioned_pairs)}')

        validate_partitioning(partitioned_pairs, batch_size)

        # Merge in the topic and content data
        # take one partition at a time and create the pytorch data

        pt('Merging partitioned_pairs into a df ... ')
        corr_df = None
        for pairs_list in tqdm(partitioned_pairs):
            if (corr_df is None):
                corr_df = dm.DataFrame(pairs_list, columns = ['t_id','c_id'])
            else:
                corr_df = dm.concat([corr_df, dm.DataFrame(pairs_list, columns = ['t_id','c_id'])], axis = 0)

        # print(corr_df)
    else:
        corr_df = pos_corr_df

    if (add_labels):
        # Generate unique integer labels for each of the t_ids
        unique_t_ids = corr_df['t_id'].unique()
        mapping_dict = {k: v for v, k in enumerate(unique_t_ids)}
        corr_df['label'] = corr_df.apply(lambda r: mapping_dict[r['t_id']], axis = 1)

    pt('Merging in topic and content data into the folds data ... ')
    df = dm.merge(corr_df , topics_df  , how='inner', left_on='t_id', right_on = 'id')
    df = df.drop(['t_id', 'id'], axis = 1)
    df = dm.merge(df      , contents_df, how='inner', left_on='c_id', right_on = 'id')
    df = df.drop(['c_id', 'id'], axis = 1)


    # t_cols = [col for col in df.columns if col.startswith("t_")]
    # c_cols = [col for col in df.columns if col.startswith("c_")]

    #pt('Converting data to torch ... ')
    #data_torch = convert_df_to_torch_no_labels(df, t_cols, c_cols)

    # return df[t_cols].copy(), df[c_cols].copy()

    if (add_labels):
        return df, len(unique_t_ids)
    return df

def get_pos_corr_subsets_for_binary_fold(fold_df, contents_df, topics_df, val = False):

    topics_subset_df   = topics_df[topics_df['id'].isin(set(fold_df['t_id'].to_list()))].copy().reset_index(drop=True).reset_index()

    if val:
        contents_subset_df = contents_df.reset_index(drop=True).reset_index()
    else:
        contents_subset_df = contents_df[contents_df['id'].isin(set(fold_df['c_id'].to_list()))].copy().reset_index(drop=True).reset_index()

    idx_corr_subset_df = pd.merge(fold_df, topics_subset_df, left_on = 't_id', right_on = 'id', how = 'inner')[['index', 'c_id']].copy().rename(columns={'index':'t_idx'})
    idx_corr_subset_df = pd.merge(idx_corr_subset_df, contents_subset_df, left_on = 'c_id', right_on ='id', how = 'inner')[['t_idx', 'index']].copy().rename(columns={'index':'c_idx'})

    corr_subset_df = idx_corr_subset_df.groupby('t_idx')['c_idx'].apply(list)
        
    return topics_subset_df, contents_subset_df, corr_subset_df


def create_train_val_split_pos_corr_from_kaggle(topics_df, correlations_df, num_splits, val_fold):

    # Split the topics which are not from "source" into 4 folds. The folds were stratified by "channels" because the hosts mentioned that the test topics could from seen channels.
    # Put the topics from fold 0 to the validation set.
    # Put all the topics from the "source" category, and the topics not from fold 0, to the training set.


    topics_df = pd.merge(topics_df[['id','category','has_content','channel']], correlations_df, left_on = 'id', right_on="topic_id", how="left")

    topics_df.drop('id',axis=1, inplace=True)


    topics_df["content_id"] = topics_df["content_ids"].str.split()
    topics_df = topics_df.explode("content_id", ignore_index=True)

    topics_df = topics_df.rename(columns={'topic_id':'t_id','content_id':'c_id'})
    
    df_train = topics_df[(topics_df["category"] == "source") & (topics_df["has_content"] == True)].reset_index(drop=True)

    print(len(df_train))
    
    df_valid = topics_df[(topics_df["category"] != "source")&(topics_df["has_content"] == True)].reset_index(drop=True)
    print(len(df_valid))


    pt(f"Splitting non-source into {num_splits} folds")
    folds = GroupKFold(num_splits)
    for i, (tr, val) in enumerate(folds.split(X=df_valid, groups=df_valid["channel"])):
        df_valid.loc[val, "fold"] = i
    print(df_valid.groupby("fold").size())
    print(df_valid[["t_id", "fold"]].drop_duplicates().groupby("fold").size())
    
    _df_train = df_valid[df_valid["fold"] != val_fold].reset_index(drop=True)
    df_valid = df_valid[df_valid["fold"] == val_fold].reset_index(drop=True)
    
    df_train = pd.concat([df_train, _df_train], ignore_index=True)

    train_df = df_train[['t_id','c_id']].copy().reset_index(drop=True)
    val_df = df_valid[['t_id','c_id']].copy().reset_index(drop=True)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
   print(torch.cuda.list_gpu_processes())

def get_labeled_data_for_pos_corr(pos_corr_df, topics_df, contents_df):
    """
    This function takes the poss corr df and returns the merged data. Each topic gets a unique label, then the topics, contents pairs are seperated and stacked.
    """

    if (isinstance(contents_df,pd.core.frame.DataFrame)):
        dm = pd
    else:
        dm = dd

    # Generate unique integer labels for each of the t_ids
    unique_t_ids = pos_corr_df['t_id'].unique()
    mapping_dict = {k: v for v, k in enumerate(unique_t_ids)}
    pos_corr_df['label'] = pos_corr_df.apply(lambda r: mapping_dict[r['t_id']], axis = 1)

    df_t = pos_corr_df[['t_id', 'label']].drop_duplicates().copy().reset_index(drop=True)
    df_c = pos_corr_df[['c_id', 'label']].drop_duplicates().copy().reset_index(drop=True)
    
    df_t.info()
    df_c.info()

    pt('Merging in topic and content data into the folds data ... ')
    df_t = dm.merge(df_t, topics_df  , how='inner', left_on='t_id', right_on = 'id')
    df_t = df_t.drop(['t_id', 'id'], axis = 1)
    df_t = df_t.rename(columns={'t_0_input_ids':'input_ids', 't_1_attention_mask':'attention_mask'})
    df_c = dm.merge(df_c, contents_df  , how='inner', left_on='c_id', right_on = 'id')
    df_c = df_c.drop(['c_id', 'id'], axis = 1)
    df_c = df_c.rename(columns={'c_0_input_ids':'input_ids', 'c_1_attention_mask':'attention_mask'})

    df = pd.concat([df_t, df_c],axis=0)

    return df, len(unique_t_ids)



import ast
import pprint

def save_dict(data_dict, filename):
    with open(filename, "w") as outfile:
        #json.dump(train_data_dict, outfile) # Json converts int keys in dicts to strings, which breaks things
        outfile.write(pprint.pformat(data_dict)) # Pretty print the dict into a text file


def load_dict(filename):
    with open(filename) as infile:
        data_dict = ast.literal_eval(infile.read()) # Read the dict from the text file
    return data_dict


def get_corr_id_pairs_df(topics_df, contents_df, corr_df):
    """
    This functions goes from topic-content index based correlation back to id based correlation

    topics_df   - data-frame with 'index' and 'id' columns
    contents_df - data-frame with 'index' and 'id' columns
    corr_df     - data-series with 't_idx' as index and 'c_idx' as a list column
    """

    corr_df = corr_df.to_frame().reset_index().explode('c_idx').reset_index(drop=True)
    pos_corr_ids_df = pd.merge(corr_df, topics_df, left_on = 't_idx', right_on = 'index', how = 'inner')[['id', 'c_idx']].copy().rename(columns={'id':'t_id'})
    pos_corr_ids_df = pd.merge(pos_corr_ids_df, contents_df, left_on = 'c_idx', right_on = 'index', how = 'inner')[['t_id', 'id']].copy().rename(columns={'id':'c_id'})
    return pos_corr_ids_df

def get_pred_corr_ids_df(topics_df, contents_df, matched_content_idxs):
    # Convert the prediction data which is a numpy array, where each row corresponds to t-idx to c-idxs prediction, to a dataframe with t_idx, c_idx pairs. 
    pairs_corr_df = pd.DataFrame(list(enumerate(matched_content_idxs.tolist())), columns = ['t_idx','c_idx']).explode('c_idx').reset_index(drop=True)

    # Some of the c_idx values might be fill values (which we have chosen to be -2), so the c-idxs correspond to only those that are 0 or more.
    pairs_corr_df = pairs_corr_df[pairs_corr_df.c_idx>=0].copy().reset_index(drop=True)

    # Convert the dataframe back to dataseries so that it can be processed by our function.
    pred_corr_idxs_ds = pairs_corr_df.groupby('t_idx')['c_idx'].apply(list)

    # Convert the dataseries with t-idx -> c-idx list to a dataframe with t-id, c-id pairs 
    pred_corr_ids_df = get_corr_id_pairs_df(topics_df, contents_df, pred_corr_idxs_ds)

    return pred_corr_ids_df


def get_corr_labeled_df( topics_df, contents_df, pos_corr_df, matched_content_idxs, add_all_pos = False):
    pos_corr_ids_df  = get_corr_id_pairs_df(topics_df, contents_df, pos_corr_df)
    pt(f'Num positive corrs: {len(pos_corr_ids_df)}')
    pred_corr_ids_df = get_pred_corr_ids_df(topics_df, contents_df, matched_content_idxs)
    pt(f'Num pred corrs: {len(pos_corr_df)}')

    intersection_corr_ids_df = pd.merge(pred_corr_ids_df, pos_corr_ids_df, how = 'inner')
    pt(f'Num correct pred corrs: {len(intersection_corr_ids_df)}')
    if (add_all_pos):
        add_corr_ids_df = pos_corr_ids_df
    else:
        # We are not adding any new corr data here, just identifying the subset of preds that are pos corr
        add_corr_ids_df          = intersection_corr_ids_df

    add_corr_ids_df['label']  = 1
    pred_corr_ids_df['label'] = 0

    # Merge the "additional" corr pairs with pred corr pairs
    corr_ids_df = pd.concat([add_corr_ids_df, pred_corr_ids_df], axis = 0).drop_duplicates(subset=['t_id', 'c_id'], keep='first').reset_index(drop=True)

    return corr_ids_df
